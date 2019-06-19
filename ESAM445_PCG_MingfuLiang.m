function ESAM445_PCG_MingfuLiang(h,p,method)
% Author: Mingfu Liang
% Date: 2019/05/07
%
% Implementation of Preconditioned Conjugate Gradient Method with different
% Relaxation scheme to solve the problem
% 
%                      -u_{xx} = f(x), u(-1)=u(1)=0 
%
%                  where       u_{exact}(x) = (1 - x^2) * exp(- p * x^2) 
%
% Input:
%          h: 
%               Input different grid size choice for the PCG method
%
%               grid size of the PCG method.
%               
%         
%          p: 
%                Input different value to determine the problem form            
%
%                parameter of problem.          
%
%          mehod:
%                Input the preconditioner for the PCG and in this problem
%                we have SSOR and Jacobi relaxtion method for
%                preconditioner, and we can also choose just CG method
%
%               If you want to choose CG, type 'CG' 
%               If you want to choose SSOR, type 'PCG-SSOR'
%               If you want to choose Jacobi, type 'PCG-Jacobi'
%                   
% Example:
%
%           ESAM445_PCG_MingfuLiang(0.01,1,'PCG-SSOR')
%           
%           means that you are going to set the grid size h=0.01 and p=1
%           to do the PCG method with SSOR as preconditioner for solving
%           the problem
%

epslion = h^2; % based on the truncation error, here for the convergence critiria 
                         % we should make sure that all the error come from the 
                         % truncation error and since the truncation error is proportional to h^2 
                         % when we are using the central difference approximation, therefore we need to 
                         % reduce the residual error to less than h^2 such that all the error come from the truncation error
                     
range=-1:h:1; % range of $x$ based on the grid size $h$
RHS = @(x) -(10*p*x.*x-2+4*p^2*x.*x-4*p^2*x.^4+-2*p).*exp(-p*x.*x); % Anonymous vectorized version of f(x)
exact_sol = @(x) (1-x.*x).*exp(-p.*x.*x); % Anonymous vectorized version of $u_{exact}$
f = RHS(range); % generate the f(x)
u_exact = exact_sol(range); % generate the $u_{exact}$
f = f'; % transpose the f(x) such that it is a column vector
u_exact = u_exact'; % % transpose the $u_{exact}$ such that it is a column vector
N = round((1-(-1))/h -1); % here I don't add one since I want to be consistent with Professor Bayliss Notebook where h= 1/(N+1) in Page 11
%method = 'PCG-SSOR'; % decide what preconditioned method you are going to use, the choices are: 'CG', 'PCG-Jacobi' and 'PCG-SSOR'
w = 2/(1+sqrt(2)*sin(pi*h)); % the optimal choice of the $\omega$ when using the SSOR

%%% test the vectorized version of f(x) and u_exact(x)

% k=1;
% for x=0:h:1
%     f_test(k)=-2*exp(-p*x^2)+8*p*(x^2)*exp(-p*x^2)+(1-x^2)*(-2*p*exp(-p*x^2)+4*(p^2)*(x^2)*exp(-p*x^2));
%     k=k+1;
% end

% k=1;
% for x=-1:h:1
%     u_exact_test(k)=(1-x^2)*exp(-p*x^2);
%     k=k+1;
% end

%%% Construct Tridiagonal Matrix A

%upper_diagonal = -1*ones(N-1);
%main_diagonal = 2*ones(N);
%lower_diagonal = -1*ones(N-1);

%%%%%%%%  un-comment them for debug propose if you want to check the matrix-form PCG %%%%%%%

% A_diag = (1/h^2)*(2*eye(N));
% A_upper_vec = (1/h^2)*(-1 *  ones(N-1,1));
% A_upper = diag(A_upper_vec,1);
% A_lower = diag(A_upper_vec,-1);
% A = A_diag + A_upper+A_lower; % define the first block matrix A1
% A_S = sparse(A); % construct the sparse form of matrix A to takes advantages of the bands of the matrix A

%%% test the efficiency about using the sparse matrix

% p_test = rand(N,1);
% tic
% A_S_p = A_S * p_test;
% elapsedTime = toc;
% fprintf('\n Using Sparse matrix need %4d seconds \n', elapsedTime);
% 
% tic
% Ap = A * p_test;
% elapsedTime = toc;
% fprintf('\n Using Non-Sparse matrix need %4d seconds \n', elapsedTime);

%%%% Preconditioned Conjugate Gradient Algorithm - PCG %%%%%%%%%
%%% All the procedure are following Professor Bayliss Note in Page 82-83 %%

x_0 = zeros(N+2,1); % initial guess of the solution x
x_k =x_0; % initialize x
r_0 = x_0; % initialize r_0
r_0(2:N+1,1) = f(2:N+1,1); % r_0 = b_0;
z_0 = x_0; % initialize z_0
z_k = z_0; % initialize z_k
r_k = r_0; % initialize r_k

if strcmp(method,'CG') % if the method just uses CG, then using the following setup for z_k
        fprintf('\n Using CG as preconditioner \n');
        z_k(2:N+1,1) =r_k(2:N+1,1);
        
        %%%%%%%%  un-comment them for debug propose if you want to check the matrix-form PCG %%%%%%%      
%         M = eye(N);
%         M = sparse(M);
%         z_k = inv(M)*r_k(2:N+1,1);
end

if strcmp(method,'PCG-Jacobi') % if the preconditioned method is Jacobi, then using the following setup for z_k
        fprintf('Using Jacobi as preconditioner');
        i = 2:N+1;
        
        %%% Since here z_k is initialized as zero before so here we did not
        %%% need to write the z_k(i-1,1) and z_k(i+1,1) again since there
        %%% are all zero here.
        
        z_k(i,1) = (1/2)* (h*h) * r_k(i,1) ;
        
        %%%%%%%%  un-comment them for debug propose if you want to check the matrix-form PCG %%%%%%% 
%     M = A_diag;
%     M = sparse(M);
%     z_k(2:N+1) = inv(M)*r_k(2:N+1,1);

end
    
if strcmp(method,'PCG-SSOR') % if the preconditioned method is SSOR, then using the following setup for z_k
    
    fprintf('Using SSOR as preconditioner');
    
    %%%% Do one SSOR and since here the z_k is initialized as zero as
    %%%% before so we just use it 
    
     %%%%%%%%% forward %%%%%%%%%%%%
        for i=2:N+1
            z_k(i,1) = (1-w)*z_k(i,1) +w*(1/2)*( z_k(i-1,1) + z_k(i+1,1) + (h*h)*r_k(i,1));
        end
    %%%%%%%%% backward %%%%%%%%%%%%
        for i=N+1:-1:2
            z_k(i,1) = (1-w)*z_k(i,1) +w*(1/2)*( z_k(i-1,1) + z_k(i+1,1)  + (h*h)*r_k(i,1));
        end
        
     %%%%%%%%  un-comment them for debug propose %%%%%%% 
%     M = (A_diag+w*A_lower)*inv(A_diag)*(A_diag+w*A_upper)/(w*(2-w));
%     M =sparse(M);
%     z_k_test = inv(M)*r_k(2:N+1,1);
end

p_1 = z_k(2:N+1,1); % initialize the p_1
p_k = p_1; % initialize the p_k
ZTR = z_k(2:N+1,1)' * r_0(2:N+1,1); % initialize the ZTR

iter_counter=0; % initialize the loop counter
while 1
    Apk=A_pk(p_k,h,N); % using the function A_pk to compute the A*pk such that 
                                     % we can take advantage of the band of the matrix A and 
                                     % only do the multiplication when the entry is non-zero in matrix A
    alpha_k =ZTR/(p_k' * Apk); % compute the $\alpha_{k}$ 
    x_k(2:N+1,1) = x_k(2:N+1,1) + alpha_k * p_k;  % update the solution x_k
    r_k(2:N+1,1) = r_k(2:N+1,1) - alpha_k * Apk;  % update the residual r_k

    %%%%%%%%  un-comment it for debug propose if you want to check the matrix-form PCG %%%%%%%
%     z_k(2:N+1,1) = inv(M)*r_k(2:N+1,1);

    if strcmp(method,'CG') % if the method is just CG, then using the following setup for z_k
        z_k(2:N+1,1) = r_k(2:N+1,1); % here since the inverse of matrix M is the identity matrix
                                                       % so we can save our memory and just use the r_k directly
    end
    
    if strcmp(method,'PCG-Jacobi') % if the preconditioned method is Jacobi, then using the following setup for z_k
        i = 2:N+1; % generate the vectorized index for parellel computation
        %%% Since here z_k is initialized as zero each time, so here we did not
        %%% need to write the z_k(i-1,1) and z_k(i+1,1) again since there
        %%% are all zero here.
        z_k(i,1) = (1/2)* (h*h) * r_k(i,1) ;
    end
    
    if strcmp(method,'PCG-SSOR') % if the preconditioned method is SSOR, then using the following setup for z_k
        %%%%%%%%% forward SOR sweep of SSOR%%%%%%%%%%%%
        z_k =zeros(N+2,1); % initialize z_k as zero each time 
        for i=2:N+1
            z_k(i,1) = (1-w)*z_k(i,1) +w*(1/2)*( z_k(i-1,1) + z_k(i+1,1) +(h*h)*r_k(i,1));
        end
    %%%%%%%%% backward SOR sweep of SSOR%%%%%%%%%%%%
        for i=N+1:-1:2
            z_k(i,1) = (1-w)*z_k(i,1) +w*(1/2)*( z_k(i-1,1) + z_k(i+1,1) + (h*h)*r_k(i,1));
        end
    end
    
    iter_counter = iter_counter+1; % update the iteration counter
    
    %%%%%%%%%%%%%% Check convergence%%%%%%%%%%%%%%%%%%%%%%
    
    norm(r_k(2:N+1,1),inf)/norm(r_0(2:N+1,1),inf) % print out the norm to know exactly what is going on for debug.
    if norm(r_k(2:N+1,1),inf)/norm(r_0(2:N+1,1),inf)<epslion
        fprintf('\n Convergence achieve. \n')
        break
    end
   
    %%%%%%%% If no convergence then continue %%%%%%%%%%%%%%%%%%%%%
    
    ZTRNEW = z_k(2:N+1,1)' * r_k(2:N+1,1); % update the ZTRNEW
    beta_k = ZTRNEW/ZTR; % update the $\beta_{k}$
    p_k = z_k(2:N+1,1) + beta_k * p_k; % update the p_k
    ZTR = ZTRNEW; % update the ZTR
    
end

%%%%%%%% Generate the visualization result of $u_{exact}$ and $x_k$ %%%
%%%%%%%% $x_k$ is the numerical approximation of the exact solution %%%

figure;
plot(range,x_k);
hold on
plot(range,u_exact,'-');
hold off
ylabel('$u(x)$','Interpreter','latex','FontSize',13)
xlabel('$x$','Interpreter','latex','FontSize',13)
leg1 = legend('$x_{k}$','$u_{exact}$');
set(leg1,'Interpreter','latex');
set(leg1,'FontSize',12);
title(['p=',num2str(p),', use ',method,' when grid size is ',num2str(h),' and iteration is ',num2str(iter_counter)],'Interpreter','latex');
end