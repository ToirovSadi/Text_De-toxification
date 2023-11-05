# Transformer

## Notebook: [transformer.ipynb](../notebooks/transformer.ipynb)

## Transformer Architecture
![Alt text](./figures/transformer1.png) <br>
Image above taken from [here](https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/49df8404d938a6edbf729876405558cc2c2b3013//assets/transformer1.png).

## Model Specification
- num_heads: 4
- hidden_dim: 256
- ff_expantion: 4
- dropout: 0.1 (encoder and decoder)
- num_layers: 3 (encoder and decoder)

## Evaluation
- You can see evaluation of all models in [this](../notebooks/evaluation.ipynb) notebook.
- You can compare all models by their bleu score (higher score better model).

## Results
- Notebooks that illustrates this architecture and results can be found [here](../notebooks/transformer.ipynb).
- Model was trained for 10 epochs.
- Sequence length is 10.
- Trained model can be found [here](../models/transformer.01.pt).

**Good Predictions** <br>
```
toxic_sent: what the hell are you talking about?
neutral_sent: what are you talking about?
prediction: what are you talking about?

toxic_sent: some kind of goddamn hero?
neutral_sent: some kind of hero?
prediction: some kind of a hero?

toxic_sent: fucked up my life.
neutral_sent: he messed up his life.
prediction: i screwed up my life.
```
**Bad Predictions** <br>
```
toxic_sent: my jelly stole my act . liar!
neutral_sent: my jelly stole my output.
prediction: my face is lying.
```

Difference between transformer and transformer2 models is that second sequence length is 32.

## Since Transformer are powerfull, let's test it with sequence length 32.

### Results
- Notebooks that illustrates this architecture and results can be found [here](../notebooks/attention2.ipynb).
- Model was trained for 10 epochs with batch_size 256.
- Trained model can be found [here](../models/attention2.01.pt).

**Good Predictions** <br>
```
toxic_sent: they said if i told anyone, they would kill zak.
neutral_sent: they said that if we told anybody that they said they would hurt zach.
prediction: they said if i told anyone, they would have killed.

toxic_sent: we are talking about a drug dealer and maybe a cop killer.
neutral_sent: we are talking about a drug dealer here and possible cop killer.
prediction: we are talking about a drug dealer and maybe a cop.
```

**Bad Predictions** <br>
```
toxic_sent: with explosives around my ankles, ready to explode.
neutral_sent: explosives still around their ankles, still ready to explode.
prediction: with my, ready to explode.
```

# Overall observation
- As you can see from the bleu scores models are not good.
- Models such as simple seq2seq had problems with memorizing context.
- Possible ways to improve the model
    1. Train dataset is too small
    2. Train dataset is not clear (lots of mistakes, misspelled words, sometimes one of sentence make no sense)
    3. Train model longer (I trained them about 10 epochs, for transformer val loss was still smaller than train loss)
- Overall task was good.
    
    

# References
- Some code and ideas were taken from this [github](https://github.com/bentrevett/pytorch-seq2seq) repository.