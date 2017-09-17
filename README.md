Meme Classifier

**How dataset structured :**

        ./data
            > ./train
                    >./meme
                    >./not_meme

            > ./validation
                    >./meme
                    >./not_meme

**Training:**

  ` python3 train.py --epochs=50 --batch_size=16 --retrain=False --weight=model_weights.h5 `

  Parameters :
  ```
    epochs     : Number of epochs                 (DEFAUT: 50)
    batch_size : Batch size                       (DEFAUT: 16)
    retrain    : Retrain model from new dataset   (DEFAUT: False)
    weight     : Weights for Retraining           (DEFAUT: model_weights.h5)
   ```

**Running/Testing:**

  ` python3 run.py --img=./data/test/maxresdefault.jpg  --threshold=0.1 --weight=model_weights.h5 `

  Parameters :
  ```
    img        : path of the image                (DEFAUT: "")
    threshold  : Threshold value for probability  (DEFAUT: 0.5)
    weight     : Weights for Retraining           (DEFAUT: model_weights.h5)
   ```

Note : Model return probability of classes by putting threshold we can predict actual classes
