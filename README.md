This is a model which trains on actor recordings, and can identify the emotional tone of speech from an audio file.

I have achieved an accuracy of over 50% with this model, running about 30 epochs, and with very little over-fitting.

Did get 70%+ test accuracy, but heavily overfitted, with 99%+ accuracy on train data. Ignored all these cases, as objective was to get a general model, which hasn't 'memorised' anything.

I could have run it for longer, as I didn't observe any plateuing, but will save that for later.

Uses pitch-shift and audio stretching to develop more training data, quintupling it.

Used 4 convolutional layers, and gap/gmp and 2 fully connected layers, and used dropout as well.

Could have modularised some of the spectrogram generation more generally, but works.
