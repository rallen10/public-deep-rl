import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

def plot_results(scores_list, window_len=100):
    """ Plot and analyze training results

    Params
    ======
        scores_list (array-like): list of score lists or list of filenames

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for scores in scores_list:
        if isinstance(scores, str):
            # load from file
            scores = pickle.load(open(scores, 'rb'))
        
        # compute moving average and standard deviation
        mv_avg = np.asarray([np.mean(scores[max(0, i-window_len):i]) for i in range(len(scores))])
        mv_std = np.asarray([np.std(scores[max(0, i-window_len):i]) for i in range(len(scores))])

        # plot
        ax.plot(np.arange(len(scores)), mv_avg)
        ax.fill_between(np.arange(len(scores)), mv_avg-mv_std, mv_avg+mv_std, alpha=0.3)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == '__main__':
    plot_results(scores_list=sys.argv[1:])