import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# code taken from: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
def display_attention(sentence, translation, attention, n_heads=4, n_rows=2, n_cols=2):
    
    assert n_rows * n_cols == n_heads
    fig = plt.figure(figsize=(15,10))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+[t.lower() for t in sentence], rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()