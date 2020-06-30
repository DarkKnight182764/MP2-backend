1. The notebook ​t.ipynb ​ this notebook contains all the code and experimentation that went into improving and iterating on the various machine learning models so that we were finally able to arrive at the novel model architecture.  
2. App.py ​ this file contains the code to start the REST api service 
3. Model_run.py ​ this file contains the code to load and run the 3 models used to make predictions. It has 2 functions predict(filepath) ​ takes in the filepath of the video on which the prediction is to be made and returns a string which is the prediction b. _get_sent_from_preds(preds) ​ is a private utility function used to condense the model predictions into a string sentence 
4. Vocab_dump ​ is a pickle file that is used by ​_get_sent_from_preds(preds) 
5. Double_attention_250e_0.12_wer ​ is the saved model that was trained for 250 epochs and resulted in a word error rate of 0..12, the training code can be found in t.ipynb
