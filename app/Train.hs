{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

import Control.Monad
import System.Environment ( getArgs )
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


iteration = 30
embed_dim = 100
hidden_dim = 100
affine_dim = 200
label_size = 3
threshold = 3


main = do
    argv <- getArgs
    (trainData : evalData : _) <- D.initialize' argv
    trainX <- T.readTrees trainData
    evalX  <- T.readTrees evalData
    let evalY = map (\(l, _, _) -> l) evalX
        vocab = T.makeVocab threshold trainX

    putStrLn $ "(embed_dim,hidden_dim,affine_dim) = " ++ show (embed_dim, hidden_dim, affine_dim)
    putStrLn $ "sizes of (trainX, evalX) = " ++ show (length trainX, length evalX)
    putStrLn $ "vocabulary size: " ++ show (length vocab)
    putStrLn $ "number of labels: " ++ show label_size

    m <- D.createModel
    pW1 <- D.addParameters' m [affine_dim, hidden_dim * 4]
    pb1 <- D.addParameters' m [affine_dim]
    pW2 <- D.addParameters' m [label_size, affine_dim]
    pb2 <- D.addParameters' m [label_size]
    trainer <- D.createAdamTrainer' m
    lstm <- T.createTreeLSTM m vocab embed_dim hidden_dim

    let batchX = T.makeBatch 500 trainX
        evalCycle = min 8 (length batchX)

    forM_ [1..iteration] $  \iter -> do
        forM_ (zip [1..] batchX) $ \(i, xs) -> do
            loss <- T.train trainer pW1 pb1 pW2 pb2 lstm xs
            D.status trainer
            print loss
            when (i `mod` evalCycle == 0) $ do
                predY <- forM evalX $ \(l, t1, t2) ->
                    D.withNewComputationGraph $ \cg -> do
                        r <- T.predict cg pW1 pb1 pW2 pb2 lstm t1 t2
                        res <- V.toList =<< D.asVector =<< D.forward cg r
                        return $ D.argmax res
                putStrLn $ "accuracy: " ++ show (T.accuracy predY evalY)
                saver <- D.createSaver' $ "models/rte_" ++ show iter ++ "_" ++ show i ++ ".model"
                D.saveModel' saver m
        D.updateEpoch trainer 1.0

