{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

import Control.Monad
import System.Environment ( getArgs )
import System.IO ( hPutStr, hPutStrLn, stderr )
import System.Exit ( die )
import qualified Data.Text as T
import qualified Data.Text.IO as T

import qualified DyNet.Core as D
import qualified DyNet.Expr as D
import qualified DyNet.RNN as D
import qualified DyNet.Dict as D
import qualified DyNet.Train as D
import qualified DyNet.IO as D
import qualified DyNet.Vector as V

import qualified TreeLSTM as T


embed_dim = 100
hidden_dim = 100
affine_dim = 200
label_size = 3
threshold = 3

argParse :: [String] -> IO (String, String, String)
argParse (modelFile : trainFile : evalFile : _) = return (modelFile, trainFile, evalFile)
argParse _ = die "predict MODEL TRAIN EVAL"

main = do
    argv <- getArgs
    (modelFile, trainFile, evalFile) <- argParse =<< D.initialize' argv
    trainX <- T.readTrees trainFile
    evalX  <- T.readTrees evalFile
    let evalY = map (\(l, _, _) -> l) evalX
        vocab = T.makeVocab threshold trainX

    m <- D.createModel
    pW1 <- D.addParameters' m [affine_dim, hidden_dim * 4]
    pb1 <- D.addParameters' m [affine_dim]
    pW2 <- D.addParameters' m [label_size, affine_dim]
    pb2 <- D.addParameters' m [label_size]
    lstm <- T.createTreeLSTM m vocab embed_dim hidden_dim

    loader <- D.createLoader modelFile
    D.populateModel' loader m
    hPutStrLn stderr $ "Model loaded from: " ++ modelFile

    predY <- forM evalX $ \(l, t1, t2) ->
        D.withNewComputationGraph $ \cg -> do
            r <- T.predict cg pW1 pb1 pW2 pb2 lstm t1 t2
            res <- V.toList =<< D.asVector =<< D.forward cg r
            let l' = D.argmax res
            T.putStrLn $ "T: " `T.append` (T.intercalate " " $ T.getTokens t1)
            T.putStrLn $ "H: " `T.append` (T.intercalate " " $ T.getTokens t2)
            putStrLn $ "GOLD: " ++ T.lshow l ++ " PREDICT: " ++ T.lshow l' ++ "\n"
            return $ l'
    hPutStrLn stderr $ "ACCURACY: " ++ show (T.accuracy predY evalY)


