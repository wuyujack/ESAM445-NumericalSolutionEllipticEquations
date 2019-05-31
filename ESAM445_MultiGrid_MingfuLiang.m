function [error_norm_vector, residual_norm_vector]=ESAM445_MultiGrid_MingfuLiang(K,N,Omega,SmoothMode,OscillatoryMode,BoundaryCondition)
% Author: Mingfu Liang
% Date: 2019/05/28
%
% Implementation of Multigrid method using weight Jacobi method for relaxation 
% to solving boundary value equation with different boundary conditions.
%
%                         - u_{xx} + K * pi * u =0
%
% Input:
%          K: 
%               Input  0 or 2 to choose different equation.
%
%               In this homewore we only have two choice of K as  0, 2. 
%         
%          N: 
%                Input the size of the grid.                
%
%                the size of the grid we are going to create. In this
%                homework we use N=64 to reproduce the numerical example in
%                Briggs' Book.
%
%          omega:
%                Input value between 0 to 1 for weight
%                   
%               The weight parameter for weight Jacobi method.
%
%          SmoothMode: 
%                The smooth mode in the initial guess.
%
%               In this homework the smooth mode in the initial guess is 16
%
%          OscillatoryMode: 
%                The oscillatory mode in the initial guess.
%
%               In this homework the oscillatory mode in the initial guess
%               is 40
%
%          BoundaryCondition:
%                Input the choice of boundary condition.
%
%               In this homework we have two the boundary condition choice:
%               
%               input 1 to choose Dirichlet conditions
%               input 2 to choose Neumann conditons
%
%  Output:
%           error_norm_vector:
%                The norm of the error vector in each step
%
%           residual_norm_vector:
%                The norm of the residual vector in each step   
%
% Example:
%
%           ESAM445_MultiGrid_MingfuLiang(0,64,2/3,16,40,1)
%           
%           means that you are going to set the grid size N=64 and omega
%           =2/3 to do the Weight Jacobi method in each relaxation step 
%           in the V-cycle in multigrid method  when K=0 and using the
%           Dirichlet conditions, which represent Eq (1) in the assignment.
%

boundary_condition = BoundaryCondition; % if boundary condition is equal to 1 means we use the Dirichlet Value 
                        % if boundary condition is equal to 2 means we use the Neumann condition
                        
w = Omega; % the parameter omega in weight Jacobi method
n = N; % the number of grid point
k1=SmoothMode; % low frequency mode in initial guess, in Book page 37
k2=OscillatoryMode; % high frequency mode in initial guess, in Book page 37
%K=2; % the parameter for the boundary value problem, it can take 0 or 2 in this assignment
res_vector_ind=1; % use to indicate the row index of the vector
err_norm_count = 2; % use to count for storaging the error vector norm in the err_vector
vector_norm_count=2; % use to count for storaging the residual vector norm in the res_norm_vector
h  = 1/n;        % grid spacing. 
u  = zeros(1,1+n);    % storage for solution, here I use n+1 since I want to be consistent with the notebook
                                        % we denote that 0,1,2,...,n, which means
                                        % that if we have n=16, we actually have n+1
                                        % points although the at 0 and n+1 it should
                                        % be zero as the boundary condition in this
                                        % Homework since we are setting at all the
                                        % boundary u=0.       
f=u; % initialize the f(x), which should be zero in this assignment
res= u;% storage for residual, here residual is a matrix
u_update  = u;         % intialize the solution update in each iteration
i=1:n+1 ; % use for parallelly generate the grid point vector 
grid_point = (i-1).*h; % generate the grid point vector
j =2:n; % the index for the parallel computation in weighted Jacobi

%%%%%%%%%%%%%%%%%%%%%
u_low = u; % initialize the low frequency mode vector
u_high =u; % initialize the high frequency mode vector
u_low(1,j)=sin(k1.*(j-1)*pi/n)/2; % generate the low frequency mode vector in a parallelism manner
u_high(1,j)=sin(k2.*(j-1)*pi/n)/2; % generate the high frequency mode vector in a parallelism manner

%%%%%%%%%%% visualize the low frequency mode and high frequency mode %%%%
figure;
plot(grid_point(2:1:end-1),u_low(2:1:end-1));
hold on;
plot(grid_point(2:1:end-1),u_high(2:1:end-1));
hold off;
legend1=['k=',num2str(k1)];
legend2=['k=',num2str(k2)];
legend(legend1,legend2);
title('visualize low frequency mode and high frequency mode');

%%%%%%%%%%% initialize the initial guess, error vector and residual vector

u(1,j)=(sin(k1.*(j-1)*pi/n)+sin(k2.*(j-1)*pi/n))/2; % initialize the initial guess

if boundary_condition ==2
    u(1,1)=0;
    u(1,2)=u(1,1);
    u(1,n+1)=u(1,n);
end

u_initial = u; % copy the initial guess for later reuse
err_inital(1,j) = 0-u_initial(1,j); % get the corresponding error vector for each individual solution
err_vector(1,1)=norm(err_inital,2); % get the norm of the initial error vector and storage it in the err_vector
res_norm = f(1,j) - (1/h^2)*( -u(1,j-1) - u(1,j+1) + (2+(h^2)*pi*pi*K)*u(1,j) ); % get the corresponding residual vector for each individual solution
res_norm_vector(1,1)= norm(res_norm,2); % get the norm of the initial residual vector and storage it in the err_vector

%%%%%%%% plot initial guess %%%%%%%%%%%%%%%%%%%%%%
figure;
plot(grid_point(2:1:end-1),u_initial(2:1:end-1));
title(['The initial guess, K=',num2str(K)]);

%%%%%%%% use weight jacobi relax three time %%%%%%%%%%%%%

for relax_time =1:3
    % use weighted Jacobi to do relaxation in fine-grid in a parallelism
    % manner
    u_update(1,j) = (1-w)*(u(1,j))+ ...
                     w*(1/(2 + (h^2)*pi*pi*K))*( u(1,j-1) + u(1,j+1) + h^2 * f(1,j));
    u = u_update; % update the approximation
    
    if boundary_condition ==2
        u(1,1)=0;
        u(1,2)=u(1,1);
        u(1,n+1)=u(1,n);
    end
    
    %%%%%%%%%%%%%%% evaluate the norm of error vector %%%%%%%%%%%%%%%%%%
    if relax_time ==1 || relax_time ==3
        err(1,j)=0-u(1,j); % get the error vector
        err_vector(res_vector_ind,err_norm_count )=norm(err,2); % storage the norm of the error vector
        err_norm_count = err_norm_count +1; %
        res_norm = f(1,j) - (1/h^2)*( -u(1,j-1) - u(1,j+1) + (2+(h^2)*pi*pi*K)*u(1,j) );
        res_norm_vector(1,vector_norm_count)= norm(res_norm,2);
        vector_norm_count = vector_norm_count + 1;
    end
    
    %%%%%%%%%%%%%% plot the top right figure in Fig. 3.5 in the book %%%%%%
    if relax_time ==1
        figure;
        plot(grid_point(2:1:end-1),u_initial(2:1:end-1),"k:");
        hold on
        plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
        hold off
        title(['K =',num2str(K),', The error after ', num2str(relax_time) ,' sweep of weighted Jacobi'])
    end
   
end

%%%%%%%%%%%%%% plot the middle left figure in Fig. 3.5 in the book %%%%%%
 figure;
 plot(grid_point(2:1:end-1),u_initial(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 hold off
 title(['K =',num2str(K),', The error after ', num2str(relax_time) ,' sweep of weighted Jacobi'])
 
 %%%%%% visualize the error with different mode in the same plot%%%%%%%%
 figure;
 plot(grid_point(2:1:end-1),u_low(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 hold off
 title(['K =',num2str(K),', The error after ', num2str(relax_time) ,' sweep of weighted Jacobi, low frequency mode'])

 figure;
 plot(grid_point(2:1:end-1),u_high(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 hold off
 title(['K =',num2str(K),', The error after ', num2str(relax_time) ,' sweep of weighted Jacobi, high frequency mode'])
 
%%%%%%%% Compute r^{2h} =I^{2h}_{h} * r^{h} %%%%%%%%%%%%
res(1,j) = f(1,j) - (1/h^2)*( -u(1,j-1) - u(1,j+1) + (2+(h^2)*pi*pi*K)*u(1,j) ); % get the corresponding residual vector for each individual solution in fine-grid
res_coarse = zeros(1,n/2+1); % initialize the coarse grid residual vector
res_coarse(1,1)=res(1,1); % for the j=0, just copy the residual from fine-grid residual vector 
res_coarse(1,end)=res(1,end); % for the j=n+1, just copy the residual from fine-grid residual vector 
res_coarse(1,2:32)=(1/4)*(res(1,2:2:end-2)+2*res(1,3:2:end-1)+res(1,4:2:end)); % interpolation fine to coarse grid, here use full weighting as a restriction operator.
e_2h = res_coarse * 0; % initialize the error vector in coarse grid
e_2h_update = e_2h; % initialize the error vector in coarse grid

%%%%%%%% Relax three times on A^{2h}*e^{2h}=r^{2h} %%%%%%%

j_2h = 2:n/2; % coarse grid index
h_c = 2*h; % coarse grid interval

for relax_time =1:3
    % use the weighted Jacobi to relax the residual equation in a
    % parallelism manner
    e_2h_update(1,j_2h) = (1-w)*(e_2h(1,j_2h))+ ...
                     w*(1/(2 + (h_c^2)*pi*pi*K))*( e_2h(1,j_2h-1) + e_2h(1,j_2h+1) + h_c^2 * res_coarse(1,j_2h)); 
    e_2h = e_2h_update; % update the error vector
    
    if boundary_condition ==2
        e_2h(1,1)=0;
        e_2h(1,2)=e_2h(1,1);
        e_2h(1,n/2+1)=e_2h(1,n/2);
    end
    %%%%%%%%%%%% check fine-grid error %%%%%%%%%%%%%%%%%%%%%%%
    e_h = res*0;
    e_h(1,1:2:end) = e_2h; % transfer back to the fine-grid
    e_h(1,2:2:end) = (e_h(1,1:2:end-2) +e_h(1,3:2:end) )/2; % transfer back to the fine-grid using interpolation
    u_new = u + e_h;
    if relax_time ==1 || relax_time ==3
        err(1,j)=0-u_new(1,j);
        err_vector(res_vector_ind,err_norm_count )=norm(err,2);
        err_norm_count = err_norm_count +1;
        res_norm = f(1,j) - (1/h^2)*( -u_new(1,j-1) - u_new(1,j+1) + (2+(h^2)*pi*pi*K)*u_new(1,j) );
        res_norm_vector(1,vector_norm_count)= norm(res_norm,2);
        vector_norm_count = vector_norm_count + 1;
    end
    
    %%%%%%%%%%%%%% plot the middle right figure in Fig. 3.5 in the book %%%%%%
    if relax_time ==1
        figure;
        plot(grid_point(2:1:end-1),u_initial(2:1:end-1),"k:");
        hold on
        plot(grid_point(2:1:end-1),u_new(2:1:end-1),"k");
        hold off
        title(['K =',num2str(K),', The fine-grid error after ', num2str(relax_time) ,' sweep of weighted Jacobi'])
    end
    
end

%%%%%%%%%%%%%% plot the bottom left figure in Fig. 3.5 in the book %%%%%%
u = u_new;
figure;
plot(grid_point(2:1:end-1),u_initial(2:1:end-1),"k:");
hold on
plot(grid_point(2:1:end-1),u_new(2:1:end-1),"k");
hold off
title(['K =',num2str(K),', The fine-grid error after ', num2str(relax_time) ,' sweep of weighted Jacobi'])

%%%%%% visualize the fine-grid error with different mode in the same plot %
 figure;
 plot(grid_point(2:1:end-1),u_low(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 hold off
 title(['K =',num2str(K),', The fine-grid error after ', num2str(relax_time) ,' sweep of weighted Jacobi, low frequency mode'])

 figure;
 plot(grid_point(2:1:end-1),u_high(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 hold off
 title(['K =',num2str(K),', The fine-grid error after ', num2str(relax_time) ,' sweep of weighted Jacobi, high frequency mode'])

%%%%%%%%%%%%%%%%%%%%%%% second iteration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

j =2:n;
u_update = zeros(1,n+1);

for relax_time =1:3
    
    u_update(1,j) = (1-w)*(u(1,j))+ ...
                     w*(1/(2 + (h^2)*pi*pi*K))*( u(1,j-1) + u(1,j+1) + h^2 * f(1,j));
    u = u_update;
    
   if boundary_condition ==2
        u(1,1)=0;
        u(1,2)=u(1,1);
        u(1,n+1)=u(1,n);
    end
    
    %%%%%%%%%%%%%%% evaluate the norm of error vector %%%%%%%%%%%%%%%%%%
    if relax_time ==3
        err(1,j)=0-u(1,j);
        err_vector(res_vector_ind,err_norm_count )=norm(err,2);
        err_norm_count = err_norm_count +1;
        res_norm = f(1,j) - (1/h^2)*( -u(1,j-1) - u(1,j+1) + (2+(h^2)*pi*pi*K)*u(1,j) );
        res_norm_vector(1,vector_norm_count)= norm(res_norm,2);
        vector_norm_count = vector_norm_count + 1;
    end
end

%%%%%%%%%%%%%% plot the bottom right figure in Fig. 3.5 in the book %%%%%%
 figure;
 plot(grid_point(2:1:end-1),u_initial(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 title(['K =',num2str(K),', The error after three sweep of weighted Jacobi, second iteration'])
 hold off
 
 %%%% visualize the fine-grid error with different mode in the same plot %%
 figure;
 plot(grid_point(2:1:end-1),u_low(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 title(['K =',num2str(K),', The error after three sweep of weighted Jacobi, second iteration, low mode'])
 hold off
 
 figure;
 plot(grid_point(2:1:end-1),u_high(2:1:end-1),"k:");
 hold on
 plot(grid_point(2:1:end-1),u(2:1:end-1),"k");
 title(['K =',num2str(K),', The error after three sweep of weighted Jacobi, second iteration, high mode'])
 hold off
 
 %%%%%%%% Compute r^{2h} =I^{2h}_{h} * r^{h} %%%%%%%%%%%%
res(1,j) = f(1,j) - (1/h^2)*( -u(1,j-1) - u(1,j+1) + (2+(h^2)*pi*pi*K)*u(1,j) ); % get the corresponding residual vector for each individual solution in fine-grid
res_coarse = zeros(1,n/2+1); % initialize the coarse grid residual vector
res_coarse(1,1)=res(1,1); % for the j=0, just copy the residual from fine-grid residual vector 
res_coarse(1,end)=res(1,end); % for the j=n+1, just copy the residual from fine-grid residual vector 
res_coarse(1,2:32)=(1/4)*(res(1,2:2:end-2)+2*res(1,3:2:end-1)+res(1,4:2:end)); % interpolation fine to coarse grid, here use full weighting as a restriction operator.
e_2h = res_coarse * 0; % initialize the error vector in coarse grid
e_2h_update = e_2h; % initialize the error vector in coarse grid

%%%%%%%% Relax three times on A^{2h}*e^{2h}=r^{2h} %%%%%%%

j_2h = 2:n/2; % coarse grid index
h_c = 2*h; % coarse grid interval

for relax_time =1:3
    % use the weighted Jacobi to relax the residual equation in a
    % parallelism manner
    e_2h_update(1,j_2h) = (1-w)*(e_2h(1,j_2h))+ ...
                     w*(1/(2 + (h_c^2)*pi*pi*K))*( e_2h(1,j_2h-1) + e_2h(1,j_2h+1) + h_c^2 * res_coarse(1,j_2h)); 
    e_2h = e_2h_update; % update the error vector
    
    if boundary_condition ==2
        e_2h(1,1)=0;
        e_2h(1,2)=e_2h(1,1);
        e_2h(1,n/2+1)=e_2h(1,n/2);
    end
    %%%%%%%%%%%% check fine-grid error %%%%%%%%%%%%%%%%%%%%%%%
    e_h = res*0;
    e_h(1,1:2:end) = e_2h; % transfer back to the fine-grid
    e_h(1,2:2:end) = (e_h(1,1:2:end-2) +e_h(1,3:2:end) )/2; % transfer back to the fine-grid using interpolation
    u_new = u + e_h;
    if relax_time ==3
        err(1,j)=0-u_new(1,j);
        err_vector(res_vector_ind,err_norm_count )=norm(err,2);
        err_norm_count = err_norm_count +1;
        res_norm = f(1,j) - (1/h^2)*( -u_new(1,j-1) - u_new(1,j+1) + (2+(h^2)*pi*pi*K)*u_new(1,j) );
        res_norm_vector(1,vector_norm_count)= norm(res_norm,2);
        vector_norm_count = vector_norm_count + 1;
    end
    
end

 %%%%% output the error reduction with respect to the initial error %%%%%
 error_reduction_to_initial_error = (err_vector(1,2:end))./(err_vector(1,1))
 error_norm_vector=err_vector;
 residual_norm_vector =res_norm_vector;
end