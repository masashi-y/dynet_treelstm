{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

module TreeLSTM (
    Token,
    Label,
    Tree(..),
    label,
    lshow,
    getTokens,
    parseLine,
    readTrees,
    TreeLSTM(..),
    createTreeLSTM,
    exprForTree,
    predict,
    train,
    makeBatch,
    accuracy,
    wordCount,
    makeVocab
) where

import Control.Monad
import Control.Arrow ( (&&&) )
import Data.List ( sort, group )
import System.Environment ( getArgs )
import qualified Data.Text as T
import qualified Data.Text.IO as T

import qualified DyNet.Core as D
import qualified DyNet.Expr as D
import qualified DyNet.RNN as D
import qualified DyNet.Dict as D
import qualified DyNet.Train as D
import qualified DyNet.Vector as V
import qualified DyNet.IO as D

-------- Some types
type Token = T.Text

type Label = Int

data Tree = Node { getLeft :: Tree, getRight :: Tree }
          | Leaf { getToken :: Token }

instance Show Tree where
    show (Node l r) = "( " ++ show l ++ " " ++ show r ++ " )"
    show (Leaf w)   = "( " ++ T.unpack w ++ " )"

label :: T.Text -> Label
label "entailment"    = 0
label "contradiction" = 1
label "neutral"       = 2

lshow :: Label -> String
lshow 0 = "entailment"
lshow 1 = "contradiction"
lshow 2 = "neutral"

getTokens :: Tree -> [Token]
getTokens (Node l r) = getTokens l ++ getTokens r
getTokens (Leaf t) = [t]

-------- Reading trees
lexer :: T.Text -> [Token]
lexer = T.split (==' ')

parser :: [Token] -> Tree
parser = head . foldl parse []
    where parse (c2 : c1 : stack) ")" = Node c1 c2 : stack
          parse stack             "(" = stack
          parse stack              t  = Leaf t : stack

parseLine :: [T.Text] -> (Label, Tree, Tree)
parseLine (l : t1 : t2 : _) = (label l, parse t1, parse t2)
    where parse = parser . lexer

readTrees :: String -> IO [(Label, Tree, Tree)]
readTrees fileName = do
    (_:contents) <- T.lines <$> T.readFile fileName
    let filtered = filter (\c -> T.head c /= '-') contents
    return $ map (parseLine . T.split (=='\t')) filtered


-------- TreeLSTM
data TreeLSTM = TreeLSTM
    {
       vocab :: D.Dict,
       pE :: D.LookupParameter,
       pWS :: [D.Parameter],
       pUS :: [D.Parameter],
       pUFS :: [D.Parameter],
       pBS :: [D.Parameter] 
    }


createTreeLSTM :: D.Model -> [Token] -> Int -> Int -> IO TreeLSTM
createTreeLSTM m vocab wdim hdim =
    TreeLSTM <$> D.createDict vocab (Just "<unk>")
             <*> D.addLookupParameters' m vocabSize [wdim]
             <*> sequence [D.addParameters' m [hdim, wdim] | _ <- "iou"]
             <*> sequence [D.addParameters' m [hdim, hdim * 2] | _ <- "iou"]
             <*> sequence [D.addParameters' m [hdim, hdim] | _ <- "ff"]
             <*> sequence [D.addParameters' m [hdim] | _ <- "iouf"]
    where vocabSize = length vocab + 1


exprForTree :: D.ComputationGraph -> TreeLSTM -> Tree -> IO (D.Expression, D.Expression)
exprForTree cg (TreeLSTM vocab pE pWS _ _ pBS) (Leaf w) = do
    emb <- D.lookup cg pE =<< D.fromString vocab w
    [_Wi, _Wo, _Wu] <- mapM (D.parameter cg) pWS
    [bi, bo, bu, _] <- mapM (D.parameter cg) pBS
    i <- D.logistic $ D.affineTransform [bi, _Wi, emb]
    o <- D.logistic $ D.affineTransform [bo, _Wo, emb]
    u <- D.tanh     $ D.affineTransform [bu, _Wu, emb]
    c <- D.cmult i u
    h <- D.cmult o (D.tanh c)
    return $ (h, c)
exprForTree cg lstm@(TreeLSTM _ _ _ pUS pUFS pBS) (Node l r) = do
    (h1, c1) <- exprForTree cg lstm l
    (h2, c2) <- exprForTree cg lstm r
    [_Ui, _Uo, _Uu] <- mapM (D.parameter cg) pUS
    [_Uf1, _Uf2] <- mapM (D.parameter cg) pUFS
    [bi, bo, bu, bf] <- mapM (D.parameter cg) pBS
    e  <- D.concat' [h1, h2]
    i  <- D.logistic $ D.affineTransform [bi, _Ui, e]
    o  <- D.logistic $ D.affineTransform [bo, _Uo, e]
    f1 <- D.logistic $ D.affineTransform [bf, _Uf1, h1]
    f2 <- D.logistic $ D.affineTransform [bf, _Uf2, h2]
    u  <- D.tanh $ D.affineTransform [bu, _Uu, e]
    c  <- D.cmult i u `D.add` D.cmult f1 c1 `D.add` D.cmult f2 c2
    h  <- D.cmult o (D.tanh c)
    return $ (h, c)


predict :: D.ComputationGraph -> D.Parameter -> D.Parameter
        -> D.Parameter -> D.Parameter
        -> TreeLSTM -> Tree -> Tree -> IO D.Expression
predict cg pW1 pb1 pW2 pb2 lstm t1 t2 = do
    _W1 <- D.parameter cg pW1
    b1 <- D.parameter cg pb1
    _W2 <- D.parameter cg pW2
    b2 <- D.parameter cg pb2
    (h1, _) <- exprForTree cg lstm t1
    (h2, _) <- exprForTree cg lstm t2
    mult <- D.cmult h1 h2
    sub <- D.abs $ D.sub h1 h2
    h <- D.concat' [h1, h2, mult, sub]
    a <- D.rectify $ D.affineTransform [b1, _W1, h]
    D.affineTransform [b2, _W2, a]


train :: D.Trainer t => t -> D.Parameter -> D.Parameter
      -> D.Parameter -> D.Parameter
      -> TreeLSTM -> [(Label, Tree, Tree)] -> IO Float
train trainer pW1 pb1 pW2 pb2 lstm ts =
    D.withNewComputationGraph $ \cg -> do
        losses <- forM ts $ \(l, t1, t2) -> do
                r <- predict cg pW1 pb1 pW2 pb2 lstm t1 t2
                D.pickneglogsoftmax r l
        lossExp <- D.sum losses
        loss <- D.asScalar =<< D.forward cg lossExp
        D.backward cg lossExp
        D.update' trainer
        return $ loss / realToFrac (length losses)


-------- Utility functions
makeBatch :: Int -> [a] -> [[a]]
makeBatch _    [] = []
makeBatch size xs = let (x, xs') = splitAt size xs in x : makeBatch size xs'

accuracy :: Eq a => [a] -> [a] -> Float
accuracy pred gold = realToFrac (length correct) / realToFrac (length pred)
    where correct = filter (\(p, g) -> p == g) $ zip pred gold

wordCount :: (Eq a, Ord a) => [a] -> [(a, Int)]
wordCount = map (head &&& length) . group . sort

makeVocab :: Int -> [(Label, Tree, Tree)] -> [Token]
makeVocab threshold ts = foldl makeVocab' [] $ wordCount $ concatMap getTokens2 ts
    where getTokens2 (_, t1, t2) = getTokens t1 ++ getTokens t2
          makeVocab' ws (w, f) = if f >= threshold then w : ws else ws
