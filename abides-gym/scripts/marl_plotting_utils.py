import time
from abides_markets.orders import Order
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scripts.marl_utils import moving_average, episode_averages, market_obs_list, mm_obs_list, pt_obs_list

colors_list = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']
markers_list = ['o','*','s','8','X','.','P','p','D']
linestyle_list = ['-','--','-.',':']
time_in_day_list = ['09:30','10:00','10:30','11:00','11:30','12:00','12:30','01:00','01:30','02:00','02:30','03:00','03:30','04:00']

def normal_density(x, cum = False):
    if cum:
        phi = 0.5 * (1 + math.erf(x / np.sqrt(2)))
    else:
        phi = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return phi

def truncated_normal_moments(mu,sigma,a,b):
    sigma = max(1e-9,sigma)
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    temp1 = (normal_density(beta) - normal_density(alpha))/(normal_density(beta,True) - normal_density(alpha,True))
    mu_ = mu - sigma * temp1
    temp2 = (beta * normal_density(beta) - alpha * normal_density(alpha))/(normal_density(beta,True) - normal_density(alpha,True))
    sigma_ = sigma**2 * (1 - temp2 - temp1**2)
    sigma_ = np.sqrt(sigma_)
    return mu_, sigma_

def load_and_plot_matching_agents(num_pts, path_list, obs_string, obs_dict_key, 
label_list, thresh = 1.6, y_lim = None):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    x_list = np.linspace(0,4+num_pts,4+num_pts)
    for path in path_list:
        idx = path_list.index(path)
        obs = np.load(path + f'{obs_string}.npy') # num_test * 5
        obs = obs/np.sum(obs,axis=1).reshape(-1,1) # convert frequency to density
        obs_avg = np.average(obs,0) # 1 * 5
        obs_std = np.std(obs, axis=0) # 1 * 5
        ax.step(x_list,obs_avg,label=label_list[idx],color=colors_list[idx],where='mid')
        ax.fill_between(x=x_list,y1=obs_avg-thresh*obs_std,
            y2=obs_avg+thresh*obs_std,alpha=0.1,color=colors_list[idx],step='mid')
    ax.set_xticks(x_list)
    ax.set_xticklabels(['Noise','Value','Momentum','MM'] + ['PT' + str(x) for x in range(1,num_pts+1)],rotation=45)
    ax.set_xlabel(f'Agent traded with {obs_dict_key}',fontsize=20,weight='bold')
    ax.set_ylabel('Frequency',fontsize=20,weight='bold')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim([obs_avg.mean()-3*obs_std.mean(),obs_avg.mean()+3*obs_std.mean()])
    ax.tick_params(axis='both',labelsize=20,pad=5)
    ax.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(path + f'{obs_string}.jpg')
    plt.close()
    return

def load_and_plot_dist_of_avg(num_pts, path_list, label_list, observables_list, obs_dict_key,
x_lim_list, order_fixed_size = None, mm_max_depth = None):
    for obs_string in observables_list:
        obs_idx = observables_list.index(obs_string)
        fig, ax = plt.subplots(1, 1, figsize=(15,6))
        n_test = np.shape(np.load(path_list[0] + f'{obs_string}.npy'))[0]
        if obs_string == 'Traded Volume' or obs_string == 'Quoted Volume' or obs_string == 'Quoted Price'\
            or (obs_string == 'Depth' and obs_dict_key == 'Market'):
            n_test = int(n_test/2)
        obs_avg = np.zeros((n_test,len(path_list)))
        for path in path_list:
            idx = path_list.index(path)
            obs = np.load(path + f'{obs_string}.npy')#[0:n_test,:] # num_test * horizon
            if obs_string == 'Traded Volume' or obs_string == 'Quoted Volume':
                obs = obs[::2,:] + obs[1::2,:] # 2num_test * horizon -> num_test * horizon
            elif obs_string == 'Quoted Price' or (obs_string == 'Depth' and obs_dict_key == 'Market'):
                obs = (obs[::2,:] + obs[1::2,:])/2 # 2num_test * horizon -> num_test * horizon
            # obs = undo_normalization(obs,obs_dict_key,obs_string,order_fixed_size,mm_max_depth)
            obs_avg[:,idx] = np.average(obs, axis=1) # num_test * 1
            if obs_dict_key == 'Principal Trader':
                label = f'PT{int(path[-2])} ' + label_list[int(idx/num_pts)]
            else:
                label = label_list[idx]
            if (obs_dict_key == 'Principal Trader' and obs_string == 'Side'):
                alpha_hist = 0.8
                ax.axvline(obs_avg[:,idx].mean(),color=colors_list[idx],linestyle='--',
                    linewidth=2,label=label)
            else:
                x_list = np.linspace(x_lim_list[obs_idx][0],x_lim_list[obs_idx][1],500)
                mu = np.average(obs_avg[:,idx])
                sigma = np.std(obs_avg[:,idx])
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x_list - mu))**2))
                scaling_factor = 1#(normal_density((x_lim_list[obs_idx][1] - mu)/sigma,True) - normal_density((x_lim_list[obs_idx][0] - mu)/sigma,True))
                y = y / scaling_factor
                ax.plot(x_list,y,color=colors_list[idx],linewidth=2)
                ax.axvline(obs_avg[:,idx].mean(),color=colors_list[idx],linestyle='--',
                    linewidth=2,label=label)
                alpha_hist = 0.2
            ax.set_xlim(x_lim_list[obs_idx])
        ax.hist(obs_avg,color=colors_list[0:len(path_list)],density=True,alpha=alpha_hist)
        ax.set_ylabel(f'Probability Density',fontsize=20,weight='bold')
        ax.set_xlabel(f'{obs_dict_key}: {obs_string}',fontsize=20,weight='bold')
        # if obs_string == 'Reward' and obs_dict_key == 'Principal Trader':
        #     ax.set_xscale('symlog')
        ax.tick_params(axis='both',labelsize=20,pad=5)
        ax.legend(title='Hyperparameters',fontsize=15,ncol=1,title_fontsize=20)
        fig.tight_layout()
        fig.savefig(path + f'{obs_string}_dist_of_avg.jpg')
        plt.close()
    return

def load_and_plot_pnl_dist(num_pts, market_path_list, mm_path_list, pt_path_list, 
label_list, order_fixed_size, mm_max_depth, x_lim_list = None):
    fig, ax = plt.subplots(int(np.ceil((1+num_pts+3)/2)), 2, figsize=(24,12))
    n_test = np.shape(np.load(mm_path_list[0] + f'Spread.npy'))[0]
    total_pnl = np.zeros((n_test,len(mm_path_list),1+num_pts+3)) 
    num_noise_agents = 20
    num_value_agents = 2
    num_momentum_agents = 2
    xlabel_list = ['MM'] + ['PT' + str(x) for x in range(1,num_pts+1)] + ['Noise','Value','Momentum']
    # [:,:,0]: MM PnL; [:,:,1]: PT1,.. [:,:,n]: PT n; [:,:,n+1]: Noise; [:,:,n+2]: Value; [:,:,n+3]: Momentum 
    lines = []
    for path in mm_path_list:
        idx = mm_path_list.index(path)
        spread_pnl = np.load(path + f'Spread PnL.npy')#[0:n_test,:] # num_test * horizon
        # inv_pnl = np.load(path + f'Inventory PnL.npy')#[0:n_test,:] # num_test * horizon
        total_pnl[:,idx,0] = np.average(spread_pnl, axis=1) # num_test * 1

        for i in range(num_pts):
            trade_pnl = np.load(pt_path_list[num_pts * idx + i] + f'Reward.npy')#[0:n_test,:] # num_test * horizon
            total_pnl[:,idx,1+i] = np.average(trade_pnl, axis=1) # num_test * 1
        
        agent_pnls = np.load(market_path_list[idx] + 'Agent PnLs.npy')
        total_pnl[:,idx,num_pts+1] = np.average(agent_pnls[:,0:num_noise_agents],axis=1) # Noise Agents
        total_pnl[:,idx,num_pts+2] = np.average(agent_pnls[:,num_noise_agents:num_noise_agents+num_value_agents],axis=1) # Value Agents
        total_pnl[:,idx,num_pts+3] = np.average(agent_pnls[:,num_noise_agents+num_value_agents:num_noise_agents+num_value_agents+num_momentum_agents],axis=1) # Momentum Agents
        
        for k in range(1 + num_pts + 3):
            i = k // 2 
            j = k % 2
            x_list = np.linspace(x_lim_list[k][0],x_lim_list[k][1],500)
            mu = np.average(total_pnl[:,idx,k])
            sigma = np.std(total_pnl[:,idx,k])
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (x_list - mu))**2))
            scaling_factor = (normal_density((x_lim_list[k][1] - mu)/sigma,True) - normal_density((x_lim_list[k][0] - mu)/sigma,True))
            y = y / scaling_factor
            # mu, sigma = truncated_normal_moments(mu,sigma,x_lim_list[k][0],x_lim_list[k][1])
            line = ax[i,j].plot(x_list,y,color=colors_list[idx],alpha=.8,linewidth=2)
            ax[i,j].axvline(total_pnl[:,idx,k].mean(),color=colors_list[idx],linestyle='--',
                linewidth=2,label=label_list[idx])
            # ax[i,j].axvline(mu,color=colors_list[idx],linestyle='--',
            #     linewidth=2,label=label_list[idx])
            alpha_hist = 0.2
            ax[i,j].set_xlim(x_lim_list[k])
            ax[i,j].hist(total_pnl[:,:,k],color=colors_list[0:len(mm_path_list)],density=True,alpha=alpha_hist)
            if j == 0:
                ax[i,j].set_ylabel(f'Probability Density',fontsize=20,weight='bold')
            ax[i,j].tick_params(axis='both',labelsize=20,pad=5)
            ax[i,j].set_xlabel(xlabel_list[k],fontsize=20,weight='bold')
        lines.append(line)
    fig.tight_layout()
    fig.text(0.83,0.8,'PnL',fontsize=40,weight='bold',
                bbox ={'facecolor':'yellow','alpha':0.6,'pad':10})
    # handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(lines,labels=label_list,title='Hyperparameters',fontsize=30,ncol=1,loc="center right",
                    borderaxespad=0.1,title_fontsize=30)
    plt.subplots_adjust(right=0.75)
    print()
    fig.savefig(market_path_list[idx] + f'PnL_dist_of_avg.jpg')
    plt.close()
    return

def load_and_training_plot(path_list, label_list, observables_list, obs_dict_key, 
window = int(1e5), max_ep = int(6e5), y_lim_list= None, min_ep = 250):
    n_cols = 3
    num_rows = int(np.ceil(len(observables_list)/n_cols))
    fig, ax = plt.subplots(num_rows, n_cols, figsize = (40,5*num_rows))
    lines = []
    for obs_string in observables_list:
        obs_idx = observables_list.index(obs_string)
        obs_idx_row = int(obs_idx / n_cols)
        obs_idx_col = obs_idx % n_cols
        for path in path_list:
            # print(f'{obs_string}: {path}')
            idx = path_list.index(path)
            obs = np.load(path + f'{obs_string}.npy') #num_train * num_iter_per_ep
            if obs_string == 'Price' and obs_dict_key == 'Market':
                num_ep = int(np.shape(obs)[0]/2)
                ask_avg = np.average(obs[0::2],axis=1) #ask prices
                bid_avg = np.average(obs[1::2],axis=1) #bid prices
                obs_avg = (ask_avg + bid_avg)/2
            elif obs_string == 'Momentum' and obs_dict_key == 'Principal Trader':
                num_ep = int(np.shape(obs)[0]/3)
                t_1_avg = np.average(obs[0::3],axis=1).reshape(-1,1) #one step momentum
                t_10_avg = np.average(obs[1::3],axis=1).reshape(-1,1) #10 steps momentum
                t_30_avg = np.average(obs[2::3],axis=1).reshape(-1,1) #30 steps momentum
                obs_avg = np.hstack((t_1_avg,t_10_avg,t_30_avg))
            else:
                obs_avg = np.average(obs,axis=1)
                num_ep = np.shape(obs)[0]
            obs_mov_avg = moving_average(obs_avg, window)
            line = ax[obs_idx_row,obs_idx_col].plot(window+np.arange(num_ep - window + 1),obs_mov_avg,
                        color=colors_list[idx % 4],marker=markers_list[idx // 4],
                        markevery=100,linewidth=6,markersize=15)
            lines.append(line)
        # ax[obs_idx_row,obs_idx_col].legend(fontsize=20,ncol=2)
        if obs_idx_row == num_rows - 1:
            ax[obs_idx_row,obs_idx_col].set_xlabel('Training Episode',fontsize=30,weight='bold')
        ax[obs_idx_row,obs_idx_col].set_ylabel(f'{obs_string}',fontsize=30,weight='bold')
        ax[obs_idx_row,obs_idx_col].tick_params(axis='both',labelsize=30,pad=10)
        ax[obs_idx_row,obs_idx_col].set_xlim([min_ep,max_ep])
        if y_lim_list is not None:
            ax[obs_idx_row,obs_idx_col].set_ylim(y_lim_list[obs_idx])
    fig.tight_layout()
    title = f'{obs_dict_key}\nobservables'
    x_pos = 0.83
    if obs_dict_key == 'Principal Trader':
        title = 'Principal Trading\n observables'
        x_pos = 0.815
    fig.text(x_pos,0.8,title,fontsize=40,weight='bold',
                bbox ={'facecolor':'yellow','alpha':0.6,'pad':10})
    fig.legend(lines,labels=label_list,title='Hyperparameters',fontsize=30,ncol=1,loc="center right",
                    borderaxespad=0.1,title_fontsize=30)
    plt.subplots_adjust(right=0.8)
    fig.savefig(path + f'{obs_dict_key}.jpg')
    plt.close()
    return

def load_and_plot_rewards(num_pts, mm_path_list, pt_path_list, label_list, obs_string, 
window = int(1e5), max_ep = int(6e5), min_ep = 250):
    fig, ax = plt.subplots(1 + num_pts, 1, figsize = (28,12))
    lines = []
    for i in range(len(mm_path_list)):
        mm_path = mm_path_list[i]
        r_mm_avg = np.average(np.load(mm_path + f'{obs_string}.npy'),axis=1) #num_train * num_iter_per_ep --> num_train
        num_ep = np.shape(r_mm_avg)[0]
        r_mm_mov_avg = moving_average(r_mm_avg, window)
        line = ax[0].plot(window+np.arange(num_ep - window + 1),r_mm_mov_avg,
                    color=colors_list[i % 4],marker=markers_list[i // 4],
                    markevery=100,linewidth=6,markersize=15)
        lines.append(line)

        for j in range(num_pts):
            pt_path = pt_path_list[num_pts * i + j]
            r_pt_avg = np.average(np.load(pt_path + f'{obs_string}.npy'),axis=1) #num_train * num_iter_per_ep --> num_train
            num_ep = np.shape(r_pt_avg)[0]
            r_pt_mov_avg = moving_average(r_pt_avg, window)
            line = ax[1 + j].plot(window+np.arange(num_ep - window + 1),r_pt_mov_avg,
                        color=colors_list[i % 4],marker=markers_list[i // 4],
                        markevery=100,linewidth=6,markersize=15)
            lines.append(line)
            ax[1 + j].set_ylabel(f'PT{j + 1}',fontsize=30,weight='bold')
            ax[1 + j].tick_params(axis='both',labelsize=30,pad=10)
            ax[1 + j].set_xlim([min_ep,max_ep])    
    ax[-1].set_xlabel('Training Episode',fontsize=30,weight='bold')
    ax[0].set_ylabel(f'MM',fontsize=30,weight='bold')
    ax[0].tick_params(axis='both',labelsize=30,pad=10)
    ax[0].set_xlim([min_ep,max_ep])
    fig.tight_layout()
    fig.text(0.83,0.7,'Rewards',fontsize=40,weight='bold',
                bbox ={'facecolor':'yellow','alpha':0.6,'pad':10})
    fig.legend(lines,labels=label_list,title='Hyperparameters',fontsize=30,ncol=1,loc="center right",
                    borderaxespad=0.1,title_fontsize=30)
    plt.subplots_adjust(right=0.72)
    fig.savefig(pt_path + f'{obs_string}.jpg')
    plt.close()
    return 

def plot_states_actions_in_market(num_pts, market_path_list, mm_path_list, pt_path_list, label_list, 
time_in_day_list, y_lim = None, order_fixed_size = None, 
mm_max_depth = None):
    fig, ax = plt.subplots(3,2,figsize=(32,15))
    temp = np.load(market_path_list[0] + 'Spread.npy')
    horizon = np.shape(temp)[1]
    n_test = np.shape(temp)[0]
    test_idx = np.random.randint(0,n_test)
    print(test_idx,n_test)
    x_list = np.linspace(0,horizon,horizon)
    lines = []
    for path in market_path_list:
        path_idx = market_path_list.index(path)

        ## Plot market states: quoted and traded prices, market spread, market depth
        quoted_prices = np.load(path + f'Quoted Price.npy')[2*test_idx:2*test_idx+2,:] # 2 * horizon
        quoted_volumes = np.load(path + f'Quoted Volume.npy')[2*test_idx:2*test_idx+2,:] # 2 * horizon
        # quoted_volumes = undo_normalization(quoted_volumes,'Market','Quoted Volume',order_fixed_size,mm_max_depth)
        # quoted_prices = undo_normalization(quoted_prices,'Market','Quoted Price',order_fixed_size,mm_max_depth)
        quoted_mid = np.average(quoted_prices,0) # horizon *
        traded_price = np.load(path + f'Traded Price.npy')[test_idx,:] # horizon *
        # print(traded_price)
        market_spread = np.load(path + f'Spread.npy')[test_idx,:]
        market_depth = np.load(path + f'Depth.npy')[2*test_idx:2*test_idx+2,:]
        # market_spread = undo_normalization(np.load(path + f'Spread.npy')[test_idx,:],
        #                     'Market','Spread',order_fixed_size,mm_max_depth) # horizon *
        # market_depth = undo_normalization(np.load(path + f'Depth.npy')[2*test_idx:2*test_idx+2,:],
        #                     'Market','Spread',order_fixed_size,mm_max_depth) # horizon *
        ax[0,0].plot(x_list,quoted_prices[0,:],label='Best Ask Price -'+label_list[path_idx],
                        linestyle=linestyle_list[0],color='r',
                        linewidth=2)
        ax[0,0].plot(x_list,quoted_prices[1,:],label='Best Bid Price -'+label_list[path_idx],
                        linestyle=linestyle_list[0],color='b',
                        linewidth=2)
        ax[0,0].plot(x_list,quoted_mid,label='Quoted Price -'+label_list[path_idx],
                        linestyle=linestyle_list[0],color=colors_list[path_idx],
                        linewidth=2)
        ax[0,0].plot(x_list,traded_price ,label='Traded Price -'+label_list[path_idx],
                        linestyle=linestyle_list[1],color=colors_list[path_idx],
                        linewidth=2)
        ax[0,0].set_ylabel(f'Price',fontsize=20,weight='bold')
        ax[0,0].legend(fontsize=10)
        ax[0,0].set_ylim([99500,100600])
        ax[0,1].plot(x_list,market_spread,label=label_list[path_idx],color=colors_list[path_idx],
                        linewidth=2)
        ax[0,1].set_ylabel(f'Market Spread',fontsize=20,weight='bold')
        ax[0,1].set_ylim([0,5])
        # ax[0,2].plot(x_list,market_depth[0,:],label='Ask Depth -'+label_list[path_idx],color=colors_list[path_idx],
        #                 linewidth=2)
        # ax[0,2].plot(x_list,market_depth[1,:],label='Bid Depth -'+label_list[path_idx],color=colors_list[path_idx],
        #                 linewidth=2)        
        # ax[0,2].set_ylabel(f'Market Depth',fontsize=20,weight='bold')
        # ax[0,0].set_ylim([80,90])

        ## Plot MM actions: half spread, depth
        mm_half_spread = np.load(mm_path_list[path_idx] + 'Spread.npy')[test_idx,:] # horizon *
        mm_depth = np.load(mm_path_list[path_idx] + 'Depth.npy')[test_idx,:] # horizon *
        ax[1,1].plot(x_list,mm_half_spread,label=label_list[path_idx],color=colors_list[path_idx],
                        linewidth=2)
        ax[1,1].set_ylabel(f'MM Half Spread',fontsize=20,weight='bold')
        ax[1,1].set_ylim([0,5])  
        
        ## Plot PT actions: distance to mid, side
        for i in range(num_pts):
            pt_dist = np.load(pt_path_list[num_pts * path_idx + i] + 'Distance to mid.npy')[test_idx,:] # horizon *
            pt_side = np.load(pt_path_list[num_pts * path_idx + i] + 'Side.npy')[test_idx,:] # horizon * in [0,1,2]
            ax[1,0].plot(x_list,pt_side,label=label_list[path_idx],color=colors_list[path_idx],
                            linewidth=2)
            ax[1,0].set_ylabel(f'PT{i+1} Order Side',fontsize=20,weight='bold')
            ax[2,1].plot(x_list,pt_dist,label=label_list[path_idx],color=colors_list[path_idx],
                            linewidth=2)
            ax[2,1].set_ylabel(f'PT{i+1} Distance to Mid',fontsize=20,weight='bold')
            ax[2,1].set_ylim([0,5])

        ax[2,0].plot(x_list,quoted_volumes[0,:],label='Best ask volume '+label_list[path_idx],color=colors_list[path_idx],
                        linewidth=2,linestyle='-') 
        ax[2,0].plot(x_list,quoted_volumes[1,:],label='Best bid volume '+label_list[path_idx],color=colors_list[path_idx],
                        linewidth=2,linestyle=':') 
        ax[2,0].legend(fontsize=20)       

    for i in range(3):
        for j in range(2): 
            ax[i,j].tick_params(axis='both',labelsize=20,pad=5)
            best_horizon = [240,270]#[0,horizon]#[150,180]
            ax[i,j].set_xlim(best_horizon)#int(2*horizon/5),int(3*horizon/5)])
            if i == 2:
                ax[i,j].set_xticks(np.arange(best_horizon[0],best_horizon[1],30))
                ax[i,j].set_xticklabels(time_in_day_list[int(best_horizon[0]/30):int(best_horizon[1]/30)])
                ax[i,j].set_xlabel(f'Time in Trading Day',fontsize=20,weight='bold')
            else:
                ax[i,j].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False) # labels along the bottom edge are off
    fig.tight_layout()
    handles, labels = ax[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Hyperparameters',fontsize=30,ncol=1,loc="center right",
                    borderaxespad=0.1,title_fontsize=30)
    plt.subplots_adjust(right=0.85)
    fig.savefig(path + f's_a_in_market.jpg')
    plt.close()
    return

def plot_market(num_pts, market_path_list, mm_path_list, pt_path_list, label_list, 
time_in_day_list, y_lim_list = None, train = False, order_fixed_size = None, 
mm_max_depth = None, mom_plot = False, plot_trades = False, show_depth = False):
    fig, ax = plt.subplots(len(market_path_list),1,figsize=(15,10))
    temp = np.load(market_path_list[0] + 'Spread.npy')
    horizon = np.shape(temp)[1]
    if train: 
        test_idx = 1500#np.shape(temp)[0] - 1
    else:
        n_test = np.shape(temp)[0]
        test_idx = 48#np.random.randint(0,n_test)
        print(test_idx,n_test)
    x_list = np.linspace(0,horizon,horizon)
    for path in market_path_list:
        path_idx = market_path_list.index(path)
        # Plot quoted and traded prices
        quoted_prices = np.load(path + f'Quoted Price.npy')[2*test_idx:2*test_idx+2,:] # 2 * horizon
        # quoted_prices = undo_normalization(quoted_prices,'Market','Quoted Price',order_fixed_size,mm_max_depth)
        quoted_mid = np.average(quoted_prices,0) # horizon *
        q_mean = 0#quoted_mid
        q_std = 1#np.std(quoted_mid)
        ax[path_idx].plot(x_list,(quoted_mid - q_mean)/q_std,label='Top of the Book',color=colors_list[0],
                        linestyle='-',linewidth=2,marker='.',markersize=10)
        ax[path_idx].fill_between(x_list,(quoted_prices[1,:] - q_mean)/q_std,(quoted_prices[0,:] - q_mean)/q_std,
                        color=colors_list[0],alpha=.4)
        
        if show_depth:
            depth = np.load(path + 'Depth.npy')[2*test_idx:2*test_idx+2,:] # 2 * horizon 
            # depth = undo_normalization(depth,'Market','Depth',order_fixed_size,mm_max_depth)
            ax[path_idx].plot(x_list,(quoted_mid + depth[0,:] - q_mean)/q_std,label='Worst Ask',
                    color=colors_list[1])
            ax[path_idx].plot(x_list,(quoted_mid - depth[1,:] - q_mean)/q_std,label='Worst Bid',
                    color=colors_list[2])
        
        traded_price = np.load(path + f'Traded Price.npy')[test_idx,:] # horizon *
        ax[path_idx].plot(x_list,(traded_price - q_mean)/q_std,label='Traded Price',color='r')
        mm_half_spread = np.load(mm_path_list[path_idx] + 'Spread.npy')[test_idx,:] # horizon *
        mm_idx = np.where((traded_price <= quoted_mid - mm_half_spread) | (traded_price >= quoted_mid + mm_half_spread))[0]
        ax[path_idx].plot(x_list,(quoted_mid - mm_half_spread - q_mean)/q_std,color=colors_list[3],
                label='MM Order')
                # label='MM Order: {:.2f}%'.format(len(mm_idx)*100/len(x_list)))
        ax[path_idx].plot(x_list,(quoted_mid + mm_half_spread - q_mean)/q_std,color=colors_list[3])
        mm_depth = np.load(mm_path_list[path_idx] + 'Depth.npy')[test_idx,:] # horizon *
        
        ax[path_idx].fill_between(x_list,(quoted_mid + mm_half_spread - q_mean)/q_std,
                                        (quoted_mid + mm_half_spread + mm_depth - 1 - q_mean)/q_std,
                                        color=colors_list[3],alpha=.4)
        ax[path_idx].fill_between(x_list,(quoted_mid - mm_half_spread - mm_depth + 1 - q_mean)/q_std,
                                        (quoted_mid - mm_half_spread - q_mean)/q_std,
                                        color=colors_list[3],alpha=.4)

        if plot_trades:
            traded_price = np.load(path + f'Traded Price.npy')[test_idx,:] # horizon *
            ax[path_idx].plot(x_list,(traded_price - q_mean)/q_std,label='Traded Price',color='r',
                            linestyle='',marker='o',markersize=8)
            mm_trades = traded_price[mm_idx]
            ax[path_idx].plot(mm_idx,(mm_trades - q_mean)/q_std,label='MM',color='m',
                    linestyle='',marker='D',markersize=7)
            value_orders = np.load(path + 'Matched Value Agent Orders.npy')[2*test_idx:2*test_idx+2,:] # 2 * horizon; 0: price, 1: side        
            value_orders[np.isnan(value_orders)] = 0
            t_idx = np.where((value_orders[0,:] > 0) & (value_orders[1,:] == 0))[0]
            ax[path_idx].plot(x_list[t_idx],(value_orders[0,:][t_idx] - q_mean)/q_std,label='Value - Sell',
                                color='b',linestyle='',marker='v',markersize=7)
            t_idx = np.where((value_orders[0,:] > 0) & (value_orders[1,:] == 2))[0]
            ax[path_idx].plot(x_list[t_idx],(value_orders[0,:][t_idx] - q_mean)/q_std,label='Value - Buy',
                                color='b',linestyle='',marker='^',markersize=7)

            pt_colors = ['c','coral','darkviolet','lime']
            for i in range(num_pts):
                pt_dist = np.load(pt_path_list[num_pts * path_idx + i] + 'Distance to mid.npy')[test_idx,:] # horizon *
                pt_side = np.load(pt_path_list[num_pts * path_idx + i] + 'Side.npy')[test_idx,:] # horizon * in [0,1,2]
                pt_idx = np.where(pt_side == 0)[0]
                ax[path_idx].plot(x_list[pt_idx],(quoted_mid[pt_idx] + pt_dist[pt_idx] - q_mean)/q_std,label=f'PT{i+1} - Sell',
                                    color=pt_colors[i],linestyle='',marker='v',markersize=10)
                pt_idx = np.where(pt_side == 2)[0]
                ax[path_idx].plot(x_list[pt_idx],(quoted_mid[pt_idx] - pt_dist[pt_idx] - q_mean)/q_std,label=f'PT{i+1} - Buy',
                                    color=pt_colors[i],linestyle='',marker='^',markersize=10)
        if y_lim_list is not None:
            ax[path_idx].set_ylim(y_lim_list[path_idx])
            ax[path_idx].set_yticks(np.linspace(y_lim_list[path_idx][0],y_lim_list[path_idx][1],5))
        ax[path_idx].set_ylabel('Price (in cents)',fontsize=20,weight='bold')
        ax[path_idx].tick_params(axis='both',labelsize=20,pad=5)
        ax[path_idx].set_title(f'{label_list[path_idx]}',fontsize=20)
        ax[path_idx].legend(fontsize=15,ncol=2)

        best_horizon = [60,90]#[0,horizon]#[210,270] #
        ax[path_idx].set_xlim(best_horizon)#int(horizon/4),int(2*horizon/4)])
        
    ax[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False) # labels along the bottom edge are off
    ax[-1].set_xticks(np.arange(best_horizon[0],best_horizon[1],30))#np.linspace(0,horizon,len(time_in_day_list)))
    ax[-1].set_xticklabels(time_in_day_list[int(best_horizon[0]/30):int(best_horizon[1]/30)])#time_in_day_list)
    ax[-1].set_xlabel(f'Time in Trading Day',fontsize=20,weight='bold')
    fig.tight_layout()
    fig.savefig(path + f'Market_snap.jpg')
    plt.close()
    return

def generate_training_plots(num_pts, mom, lam, vol_mm_list, vol_pt_list, L_list, d_list,
window = 500, max_ep = 2000, min_ep = 500):
    print('Plotting Training analysis')
    start_time = time.time()
    label_list = []
    market_path_list = []
    mm_path_list = []
    pt_path_list = []
    for d in d_list:
        for L in L_list:
            M = L
            for v1 in vol_mm_list:
                v2 = v1
                # label_list.append(f'Volume: {v1}; L: {L}; M: {M}; $\delta$: {d}\n $\lambda_a$: {lam}, Mom: {mom}')
                label_list.append(f'Volume: {v1}; L: {L}; $\delta$: {d}')
                name_xp = f"ppo_marl_affairs_vol{v1}{v2}_L{L}_d{d}_M{M}_pts{num_pts}_momentum_agent_freq{mom}_lambda_a{lam}/"
                log_dir = os.getcwd() + '/results/' + name_xp + '/'
                market_path_list.append(log_dir+'Market/')
                mm_path_list.append(log_dir+'MM/')
                for i in range(num_pts):
                    pt_path_list.append(log_dir+f'PT{i+1}/')
    # Plot rewards to look for convergence
    load_and_training_plot(market_path_list,label_list,market_obs_list,'Market',
                    window,max_ep,min_ep=min_ep)
    load_and_training_plot(mm_path_list,label_list,mm_obs_list[:-1],
                    'Market Maker',window,max_ep,min_ep=min_ep)
    load_and_training_plot(pt_path_list,label_list,pt_obs_list[:-1],
                    'Principal Trader',window,max_ep,min_ep=min_ep)
    load_and_plot_rewards(num_pts, mm_path_list, pt_path_list, label_list, 'Reward', 
                    window, max_ep, min_ep)

    # Plot traders that trade with our learning agents
    load_and_plot_matching_agents(num_pts,mm_path_list,'Matching Agents','Market Maker',label_list,y_lim=(0,1))
    for i in range(num_pts):
        load_and_plot_matching_agents(num_pts,pt_path_list[i::num_pts],'Matching Agents','Principal Trader',
            label_list,y_lim=(0,1))

    # Plot snapshot of market in last episode
    plot_market(num_pts,market_path_list,mm_path_list,
            pt_path_list,label_list,
            time_in_day_list,None,#[(100075,100110),(99900,99940)],
            True,plot_trades=True)
    print(f'Time to plot: {(time.time() - start_time)/60} minutes')    
    return

def generate_test_plots(num_pts, mom, lam, vol_mm_list, vol_pt_list, L_list, d_list, 
plot_dist = True, order_fixed_size = None, mm_max_depth = None, thresh = 1.6):
    print('Plotting Test analysis')
    start_time = time.time()
    label_list = []
    market_path_list = []
    mm_path_list = []
    pt_path_list = []
    for d in d_list:
        for L in L_list:  
            M = L  
            for v1 in vol_mm_list:
                v2 = v1
                label_list.append(f'Volume: {v1}; L: {L}; M: {M}; $\delta$: {d}')
                name_xp = f"ppo_marl_affairs_vol{v1}{v2}_L{L}_d{d}_M{M}_pts{num_pts}_momentum_agent_freq{mom}_lambda_a{lam}/"
                log_dir = os.getcwd() + '/results/test/' + name_xp + '/'
                market_path_list.append(log_dir+'Market/')
                mm_path_list.append(log_dir+'MM/')
                for i in range(num_pts):
                    pt_path_list.append(log_dir+f'PT{i+1}/')

    market_lim_list = [(95000,104000),(80,160),(95000,104000),(8,30), #['Quoted Price','Quoted Volume','Traded Price','Traded Volume'
                        (0,200),(0,4000)] #,'Spread','Depth']
    mm_lim_list = [(-2500,2000),(-2e8,2.5e8),(0.5,2.6),(1.5,5), #['Inventory','Cash','Spread','Depth'
                    (3,25),(-7.5e4,7.5e4),(-7.5e4,7.5e4)] #'Spread PnL','Inventory PnL','Reward']
    pt_lim_list = [(-600,1200),(-12e7,6e7),(0.8,1.6),(0,2), #['Inventory','Cash','Distance to mid','Side',
                    (-750,500)] #,'Reward'... ,'Momentum','Price History']
    
    # Plot traders that trade with our learning agents
    load_and_plot_matching_agents(num_pts,mm_path_list,'Matching Agents','Market Maker',label_list,y_lim=(0,1))
    for i in range(num_pts):
        load_and_plot_matching_agents(num_pts,pt_path_list[i::num_pts],'Matching Agents','Principal Trader',
                    label_list,y_lim=(0,1))

    # Plot snapshot of market in sample test episode
    plot_market(num_pts,market_path_list,mm_path_list,pt_path_list,
                label_list,time_in_day_list,None,#[(99835,99850),(100080,100100)],
                order_fixed_size=order_fixed_size,
                mm_max_depth=mm_max_depth,plot_trades=True,show_depth=False)
    # plot_states_actions_in_market(num_pts,market_path_list,mm_path_list,pt_path_list,label_list,time_in_day_list,
    #     None,order_fixed_size=order_fixed_size,mm_max_depth=mm_max_depth)

    if plot_dist:
        load_and_plot_dist_of_avg(num_pts,market_path_list,label_list,market_obs_list[:-1],'Market',
            market_lim_list,order_fixed_size,mm_max_depth)
        load_and_plot_dist_of_avg(num_pts,mm_path_list,label_list,mm_obs_list[:-1],
            'Market Maker',mm_lim_list,order_fixed_size,mm_max_depth)
        load_and_plot_dist_of_avg(num_pts,pt_path_list,label_list,pt_obs_list[:-3],
            'Principal Trader',pt_lim_list,order_fixed_size,mm_max_depth)  
        load_and_plot_pnl_dist(num_pts,market_path_list,mm_path_list,pt_path_list,
            label_list,order_fixed_size,mm_max_depth,
            [(5,25),(-750,500),(-4e5,3e5),(-1e7,1.5e7),(-.7e6,1e6)])
    print(f'Time to plot: {(time.time() - start_time)/60} minutes')    
    return