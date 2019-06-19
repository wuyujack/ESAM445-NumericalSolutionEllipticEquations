function Apk=A_pk(p_k,h,N)
% Author: Mingfu Liang
% Date: 2019/05/07
%
% Compute the multiplication between the tridiagonal matrix A and column
% vector p_k without computing the zero element in matrix A and only take
% advantage of the bands of matrix A.
% 
% Input:
%          p_k: 
%               The column vector p_k
%         
%          h: 
%                The grid size h           
%
%          N:
%                The dimension of the matrix A
%                   
% Output:
%         Apk:
%                The multiplication of A * p_k, the dimension of Apk
%                should be N * 1


            Apk(1,1)=(1/h^2)*(2*p_k(1,1) - p_k(2,1)); % the first entry of the multiplication
            Apk(2:N-1,1)=(1/h^2)*(-p_k(1:end-2,1)+2*p_k(2:end-1,1)-p_k(3:end,1)); % the second to the last but one entry
            Apk(N,1)=(1/h^2)*(-p_k(end-1,1) +2 * p_k(end,1)); % the last entry of the multiplication
end