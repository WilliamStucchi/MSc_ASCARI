%Nathan Spielberg 2019
%Plot Science Robotics Experimental Data
%For Comparison between FFW FB Controller with Bike Model
%And NN based steady state FFW FB Controller
close all
clc
clear all

%First Load in the Experimental data
load('Bike_Exp.mat');
load('NN_Exp.mat');

bike_color     = [166/255 206/255 227/255];
bike_color_dev = [31/255 120/255 180/255];
nn_color       = [178/255 223/255 138/255];
nn_color_dev   = [51/255 160/255 44/255];
comp_color     = [152,78,163]/255;

CL_WIDTH = 2.5

%First Plot the track map in North and East Coordinates
figure()
%figure('Renderer', 'painters', 'Position', [10 10 6000 600])
%plot(Nor1(min_sref2(i_nn-1):min_sref2(i_nn)-1), Eas1(min_sref2(i_nn-1):min_sref2(i_nn)-1) ,'k', 'LineWidth', 3.5)
plot(N, E ,'k', 'LineWidth', 3.5)
axis equal
xlabel('East (m)')
ylabel('North (m)')
xlim([-480 -370])
set(gca, 'FontSize', 7, 'FontName', 'Helvetica')
x0=0;
y0=0;
width=2.92;
height=2.27;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Now Plot Comparisons Panel
figure()
subplot(3,1,1)
plot(s_lk, e_lk,'Color',bike_color_dev, 'linewidth', 2)
hold on
plot(s, e,'Color',nn_color_dev, 'linewidth', 2)
grid on
legend('Physics Model', 'Neural Network')
ylabel('e (m)')
set(gca, 'FontSize', 7, 'FontName', 'Helvetica')
x0=4;
y0=4;
width=3.1;
height=3.53;
set(gcf,'units','inches','position',[x0,y0,width,height])
% does this actually set width and height?

%Now Plot Comparison of Steering Angle
subplot(3,1,2)
plot(s_lk, Delta_rw_lk*180.0/pi,'Color',bike_color_dev, 'linewidth', 2)
hold on
plot(s, Delta_rw*180.0/pi,'Color',nn_color_dev, 'linewidth', 2)
grid on
ylabel('Steering Angle (deg)')
set(gca, 'FontSize', 7, 'FontName', 'Helvetica')

%Finally Plot Steering Angle Difference Between Two Controllers
subplot(3,1,3)
%Note this is approximate as the tests are slighly at different speeds
%because of imperfect speed tracking, therefore the laps are compared from
%start to end of the smallest sample size lap. The total distance travelled
%between both tests is equal.
Steering_Difference = Delta_rw- Delta_rw_lk(1:end-12)
plot(s, Steering_Difference*180.0/pi  ,'Color', comp_color, 'linewidth', 2)
grid on
xlabel('s (m)')
ylabel('Difference (deg)')
set(gca, 'FontSize', 7, 'FontName', 'Helvetica')

%histogram comparison
n_bins = 25
figure()
h1 = histogram(e_lk, n_bins,'FaceColor', bike_color, 'BinLimits',[-0.5 0.1])
hold on
h2 = histogram(e, n_bins, 'FaceColor', nn_color, 'BinLimits',[-0.5 0.1])
xlabel('Tracking Error (m) ')
ylabel('Counts')
%title('Comparison of Tracking Performance')
legend('Physics Model','Neural Network')
set(gca, 'FontSize', 7, 'FontName', 'Helvetica')

x0=2;
y0=2;
width=2.79;
height=2.92;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Supplementary Plots

%Comparison of Predicted and Measured Side Slip Angle Beta
figure()
plot(s_lk, Beta_measured_lk*180.0/pi, 'Color', bike_color, 'linewidth', 2)
grid on
hold on
plot(s, Beta_measured*180.0/pi,'Color', nn_color, 'linewidth', 2)
hold on
plot(s_lk, Beta_ffw_lk*180.0/pi,'Color',bike_color_dev, 'linewidth', 2)
hold on
plot(s, Beta_ffw*180.0/pi,'Color',nn_color_dev , 'linewidth', 2)
legend('Physics Meas', 'NN Meas', 'Physics Pred', 'NN Pred')
xlabel('s (m)')
ylabel('Sideslip (deg)')
set(gca, 'FontSize', 7, 'FontName', 'Arial')
x0=2;
y0=2;
width=3.5;
height=3.5;
set(gcf,'units','inches','position',[x0,y0,width,height])

figure()
plot(s_lk, Ux_lk,'Color',bike_color_dev, 'linewidth', 2)
hold on
plot(s, Ux,'Color',nn_color_dev, 'linewidth', 2)
grid on
legend('Physics Model', 'NN Model')
xlabel('s (m)')
ylabel('Vx (m/s)')
set(gca, 'FontSize', 7, 'FontName', 'Arial')
x0=2;
y0=2;
width=3.5;
height=3.5;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Plot the Objective Value of the Neural Network optimization during the testing.
figure()
plot(s, Obj_val,'Color',nn_color_dev, 'linewidth', 2)
ylabel('Objective Value')
xlabel('s (m)')
set(gca, 'FontSize', 7, 'FontName', 'Arial')
x0=2;
y0=2;
width=3.5;
height=3.5;
set(gcf,'units','inches','position',[x0,y0,width,height])

Mean_Obj_val = mean(Obj_val)
Max_Obj_val  = max(Obj_val)
Min_Obj_val  = min(Obj_val)




