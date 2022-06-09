from scripts.marl_plotting_utils import generate_training_plots, generate_test_plots

train_plots = 1
test_plots = 1
plot_dist = 1
vol_mm_list = [0,1]
vol_pt_list = [0,1]
L_list = [6]
d_list = [0]
mom_freq_list = ["5min"]#, "5min"]
lambda_a_list = [5.7e-12]#,2.85e-12]
order_fixed_size = 20
mm_max_depth = 5
num_pts = 1

for m in mom_freq_list:
        for lam in lambda_a_list:
            # print(m,lam)
            if train_plots:
                generate_training_plots(num_pts,m,lam,vol_mm_list,vol_pt_list,L_list,d_list)

            if test_plots:
                generate_test_plots(num_pts,m,lam,vol_mm_list,vol_pt_list,L_list,d_list,
                    plot_dist,order_fixed_size,mm_max_depth)