function ESAM445_SOR_MingfuLiang(K,N,omega,starting_method)
% Author: Mingfu Liang
% Date: 2019/05/07
%
% Implementation of SOR Method for solving Helmholtz equation.
%
% Input:
%          K: 
%               Input -2, 0 or 2 to choose different Helmholtz equation.
%
%               the parameter K in the Helmholtz equation. In this homework
%               we only have three choice of K as -2, 0, 2. 
%         
%          N: 
%                Input 16, 32 or 64 to choose the grid size.                
%
%                the size of the grid we are going to create. In this
%                homework we have three alternative choice which is N=16
%                N=32 and N=64.
%          omega:
%                Input value between 1 to 2 for weight since
%                overrelaxation.
%                   
%               The weight parameter for SOR method.
%
%          starting_method: 
%                Input 1 or 2 to choose the starting method.
%
%                the choice of the starting data we are going to use. In
%                this homework we have two type of starting method, which
%                are:
%                      (a) u_{i,j} = 1
%                      (b) u_{i,j} = (-1)^{i+j}
%                 To use the starting method (a), please input 1.
%                 To use the starting method (b), please input 2.
% 
% Example:
%
%           ESAM445_SOR_MingfuLiang(-2,16,1.8,1) 
%           
%           means that you are going to set the grid size N=16 and omega
%           =1.8 to do the SOR method for Helmholtz equation 
%           when K=-2 and using the starting data (a) u_{i,j}=1.
%

tic
%%%  initialize parameter
w = omega;
nonhom = @(X,Y) 32.*X.*Y.*(X-1).*(1-Y); % define the nonhomogenous part of the equation, which is the right handside 
h     = 1/(N+1);        % grid spacing. Here I use N+1 since I want to be consistent with the notebook
tol   = 1e-7;        % tolerance

u   = zeros(N+2,N+2);    % storage for solution, here I use N+2 since I want to be consistent with the notebook
                                        % we denote that 0,1,2,...,N,N+1, which means
                                        % that if we have N=16, we actually have N+2
                                        % points although the at 0 and N+1 it should
                                        % be zero as the boundary condition in this
                                        % Homework since we are setting at all the
                                        % boundary u=0.
                                        
res     = u;% storage for residual, here residual is a matrix
loop_count     = 0;        % while loop counter
[X,Y] = meshgrid(0:h:1,0:h:1); % coordinates for final solution generation.
f     = nonhom(X,Y); % generate the nonhomogenous part matrix 

%%%

%Now start the iteration solver, stop when
%infinite norm of the residual vector < tolerance

figure;

%%%% vectorized the iteration, generate the index set such that all the calculation can done simultaneous %%%%%%%%%% 

%%%% initial vectorize index for generating intial guess of starting points %%%%

l =2:N+1;
m=2:N+1;

%%%% initial guess of starting points %%%%

if starting_method ==1
    u(l,m)=1;
    starting_name = '(a)';
end

if starting_method ==2
    for i =2:N+1
        for j =2:N+1
            u(i,j)=(-1)^(i+j);
        end
    end
    starting_name = '(b)';
end

%%%% generate the index to do vectorize calculation later

i = 2:N+1; % start from 2 and end at N+1 since I don't want to update the boundary value since they should be zero all the time
j = 2:N+1; % start from 2 and end at N+1 since I don't want to update the boundary value since they should be zero all the time

%%%% initial the boundary value %%%%
%%%% In this homework we set u=0 at all the boundary %%%%

    u(1,:) =0;
    u(:,1) =0;
    u(N+2,:) =0;
    u(:,N+2) =0;     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while 1

    %%% SOR Implementation. Here in SOR, the update of each individual
    %%% element in solution matrix is dependent with the previous update.
    %%% Here I use the Forward Euler to get the general update formula
    %%% for the u(i,j). The outer loop is in i and inner loop is in j,
    %%% which mean the outer loop is for row loop and inner loop is for
    %%% column loop. For example for i=1, which means for the first row, 
    %%% we loop j=1:N+1, which means loop for all the columns in first row.
    %%% For example for i=1, j=1, we first update u(1,1), then i is still 1
    %%% and j become 2, and since I just use the same u matrix to storage
    %%% the solution, therefore when calculate u(1,2) by the fomular below,
    %%% it automatically use the updated u(1,1) by the previous iteration. So
    %%% I save the memory by just using one matrix to storage and update
    %%% the solution and also satisfies the requirement of SOR.
        
    for i=2:N+1
        for j=2:N+1
            u(i,j) = (1-w)*u(i,j) +w*(1/(4 - (h^2)*pi*pi*K))*( u(i,j+1) + u(i,j-1) +u(i-1,j) + u(i+1,j) -h^2*f(i,j));
        end
    end
    
    
    res(l,m) = f(l,m) - (1/h^2)*( u(l-1,m) + u(l+1,m) + u(l,m-1) + u(l,m+1) - (4-(h^2)*pi*pi*K)*u(l,m) ); % get the corresponding residual matrix for each individual solution
    loop_count = loop_count+1; % update the iteration count of loop
    
    %%% convergence check using infinite vector norm of residue matrix
    %%% be careful here you should first use res(:) to change the matrix to
    %%% a vector so that you can use vector norm correspondingly.
    
    if norm(res(:),inf) < tol
         break
    end
    
end

%%% plot the solution at convergence iteration

mesh(X,Y,u);
title1 = ['K=', num2str(K), ',omega=',num2str(w),',N=',num2str(N), ',SOR Solution at ',num2str(loop_count),' iteration, starting data ', starting_name];
title(title1)
toc
end