import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import torch
# from unityagents import UnityEnvironment
# from dqn_brain_agent import train_dqn, BrainAgent

# def visualize_agent(checkpoint_filename='default_checkpoint.pth'):
#     """ Render agent interacting with environment
#     """

#     env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
#     brain_agent = BrainAgent(env.brain_names[0], env.brains[env.brain_names[0]])
#     brain_agent.dqn_local.load_state_dict(torch.load(checkpoint_filename))
#     train_dqn(env, brain_agent, train_mode=False, n_episodes=1, eps_start=0., eps_end=0.)
#     env.close()


def plot_actor_critic_results(algorithm_results_list, threshold=None, window_len=100, 
                plt_title=None):
    """ Plot and analyze training results from actor critic methods

    Params
    ======
        scores_list (array-like): list of score lists or list of filenames

    """

    # extract data
    scores_list = []
    pol_loss_list = []
    val_loss_list = []
    clipped_L_list= []
    entropy_list = []
    alg_titles = []

    for alg_res in algorithm_results_list:
        if isinstance(alg_res, str):
            # load from file
            alg_titles.append(alg_res)
            data = pickle.load(open(alg_res, 'rb'))
        scores_list.append(data['scores'])
        pol_loss_list.append(data['policy_loss'])
        val_loss_list.append(data['value_loss'])
        clipped_L_list.append(data['clipped_surrogate'])
        entropy_list.append(data['entropy'])

    # plot scores
    fig = plt.figure("scores")
    ax = fig.add_subplot(111)

    for scores in scores_list:
        
        # compute moving average and standard deviation
        mv_avg = np.asarray([np.mean(scores[max(0, i-window_len):i]) for i in range(len(scores))])
        # mv_std = np.asarray([np.std(scores[max(0, i-window_len):i]) for i in range(len(scores))])
        mv_q16 = np.asarray([np.quantile(scores[max(0, i-window_len):i], 0.16) for i in range(1,len(scores))])
        mv_q84 = np.asarray([np.quantile(scores[max(0, i-window_len):i], 0.84) for i in range(1,len(scores))])
        mv_q16 = np.insert(mv_q16, 0, 0.0)
        mv_q84 = np.insert(mv_q84, 0, 0.0)


        # plot
        ax.plot(np.arange(len(scores)), mv_avg)
        # ax.fill_between(np.arange(len(scores)), mv_avg-mv_std, mv_avg+mv_std, alpha=0.3)
        ax.fill_between(np.arange(len(scores)), mv_q16, mv_q84, alpha=0.3)

    # plot success threshold
    if threshold is not None:
        plt.hlines(threshold, 0, len(scores), colors='r', linestyles='dashed')
    plt.title(plt_title)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(alg_titles)


    # plot losses
    fig = plt.figure("losses")
    ax = fig.add_subplot(111)
    for pol_losses in pol_loss_list:
        
        # # compute moving average and standard deviation
        # mv_avg = np.asarray([np.mean(pol_losses[max(0, i-window_len):i]) for i in range(len(pol_losses))])
        # # mv_std = np.asarray([np.std(pol_losses[max(0, i-window_len):i]) for i in range(len(pol_losses))])
        # mv_q16 = np.asarray([np.quantile(pol_losses[max(0, i-window_len):i], 0.16) for i in range(1,len(pol_losses))])
        # mv_q84 = np.asarray([np.quantile(pol_losses[max(0, i-window_len):i], 0.84) for i in range(1,len(pol_losses))])
        # mv_q16 = np.insert(mv_q16, 0, 0.0)
        # mv_q84 = np.insert(mv_q84, 0, 0.0)


        # plot
        ax.plot(np.arange(len(pol_losses)), pol_losses)
        # ax.fill_between(np.arange(len(pol_losses)), mv_avg-mv_std, mv_avg+mv_std, alpha=0.3)
        # ax.fill_between(np.arange(len(pol_losses)), mv_q16, mv_q84, alpha=0.3)

    for clipped_L in clipped_L_list:
        
        # plot
        ax.plot(np.arange(len(clipped_L)), clipped_L)

    for entropy in entropy_list:
        ax.plot(np.arange(len(entropy)), entropy)


    for val_losses in val_loss_list:
        
        # plot
        ax.plot(np.arange(len(val_losses)), val_losses)


    # plot success threshold
    if plt_title is not None:
        plt.title(plt_title + ": losses")
    plt.ylabel('Losses')
    plt.xlabel('Training Iteration #')
    plt.legend(['policy loss', 'clipped surrogat', 'entropy', 'value loss'])

    # open plots
    plt.show()

if __name__ == '__main__':
    plot_actor_critic_results(algorithm_results_list=sys.argv[1:])
