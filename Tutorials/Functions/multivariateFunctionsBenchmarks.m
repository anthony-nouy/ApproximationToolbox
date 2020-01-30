% Copyright (c) 2020, Anthony Nouy, Erwan Grelier, Loic Giraldi
% 
% This file is part of ApproximationToolbox.
% 
% ApproximationToolbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% ApproximationToolbox is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
% 
% You should have received a copy of the GNU Lesser General Public License
% along with ApproximationToolbox.  If not, see <https://www.gnu.org/licenses/>.

function [fun,RV] = multivariateFunctionsBenchmarks(cas,varargin)
% fun = multivariateFunctionsBenchmarks(functionName,varargin);
% functionName: Name of the function, varargin : parameters of the function
% fun: handle function
% RV: input random variables
%
% functionName = 'borehole', 'ishigami', 'sinofasum', 'linearadditive', 'linearrankone',
% 'quadraticrankone', 'orange', 'canonicalrank2', 'mixture', 'field',
% 'oscillatory','productpeak','cornerpeak','gaussian','continuous','discontinuous',
% 'henonheiles', 'sobol', 'anisotropic', 'polynomial', 'geometricbrownian'

switch lower(cas)
    case 'borehole'
        X = cell(8,1);
        
        X{1}=NormalRandomVariable(0.1,0.0161812);
        X{2}=NormalRandomVariable(0,1);
        X{3}=UniformRandomVariable(63070,115600);
        X{4}=UniformRandomVariable(990,1110);
        X{5}=UniformRandomVariable(63.1,116);
        X{6}=UniformRandomVariable(700,820);
        X{7}=UniformRandomVariable(1120,1680);
        X{8}=UniformRandomVariable(9855,12045);
        
        X = RandomVector(X);
        RV = getStandardRandomVector(X);
        
        g = @(x) 2*pi*x(:,3).*(x(:,4)-x(:,6))./(...
            log(exp(7.71+1.0056*x(:,2))./x(:,1)).*...
            (1+2*x(:,7).*x(:,3)./log(exp(7.71+1.0056*x(:,2))./x(:,1))./x(:,1).^2./x(:,8)+...
            x(:,3)./x(:,5)));
        fun = @(x) g(transfer(RV,X,x));
    case 'ishigami'
        % fun(x) = sin(x_1) + a*sin(x_2)^2 + b*(x_3)^4*sin(x_1)
        % (by default a=7, b=0.1)
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        if nargin<=2
            a = 7;
        else
            a = varargin{2};
        end
        if nargin<=3
            b = 0.1;
        else
            b = varargin{3};
        end
        fun = @(x) sin(x(:,1)) + a.*sin(x(:,2)).^2 + b.*x(:,3).^4.*sin(x(:,1));
        RV = RandomVector(UniformRandomVariable(-pi,pi),d);
        % fun = @(x) sin(pi*x(:,1)) + a.*sin(pi*x(:,2)).^2 + b.*(pi*x(:,3)).^4.*sin(pi*x(:,1));
        % RV = RandomVector(UniformRandomVariable(-1,1),d);
        % fun = @(x) sin(-pi+2*pi*x(:,1)) + a.*sin(-pi+2*pi*x(:,2)).^2 + b.*(-pi+2*pi*x(:,3)).^4.*sin(-pi+2*pi*x(:,1));
        % RV = RandomVector(UniformRandomVariable(0,1),d);
    case 'sinofasum'
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        if nargin<=2
            fun = @(x) sin(sum(x,2));
        else
            fun = @(x) sin(x*varargin{2}(:));
        end
        RV = RandomVector(UniformRandomVariable(),d);
    case 'linearadditive'
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        if nargin<=2
            fun = @(x) sum(x,2);
        else
            fun = @(x) x*varargin{2}(:);
        end
        RV = RandomVector(UniformRandomVariable(),d);
    case 'linearrankone'
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        if nargin<=2
            w = zeros(1,d);
        else
            w = varargin{2};
        end
        fun = @(x) prod(repmat(w(:)',size(x,1),1)+2*x,2);
        RV = RandomVector(UniformRandomVariable(),d);
    case 'quadraticrankone'
        d = varargin{1};
        c = varargin{2};
        fun = @(x) prod(repmat(c(1,:),size(x,1),1)+...
            repmat(c(2,:),size(x,1),1).*x+...
            repmat(c(3,:),size(x,1),1).*x.^2,2);
        RV = RandomVector(UniformRandomVariable(),d);
    case 'orange'
        if nargin<=1
            d = 4;
        else
            d = varargin{1};
        end
        fun = @orange;
        RV = RandomVector(UniformRandomVariable(),d);
    case 'canonicalrank2'
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        fun = @(x) x(:,1).*x(:,2).*x(:,3)+x(:,1).^2 + x(:,2);
        RV = RandomVector(UniformRandomVariable(),d);
    case 'mixture'
        if nargin<=1
            d = 6;
        else
            d = varargin{1};
        end
        fun = @(x) sin(x(:,1)+x(:,4)).*exp(x(:,5)).*x(:,6) + sin(x(:,3).*x(:,4)).*x(:,6);
        RV = RandomVector(UniformRandomVariable(),d);
    case 'field'
        if nargin<=1
            d = 6;
        else
            d = varargin{1};
        end
        fun = @(x) 1 + cos(x(:,1)).*x(:,2) + sin(x(:,1)).*x(:,3) + exp(x(:,1)) .* x(:,4) + 1./(x(:,1)+1).*x(:,5) + 1./(2*x(:,1)+3).*x(:,6);
        RV = RandomVector(UniformRandomVariable(0,1),d);
    case {'oscillatory','productpeak','cornerpeak','gaussian','continuous','discontinuous'}
        if nargin<=1
            d = 10;
        else
            d = varargin{1};
        end
        rng(1)
        w = rand(1,d);
        rng(2);
        c = rand(1,d);
        rng('default');
        RV = RandomVector(UniformRandomVariable(0,1),d);
        switch cas
            case 'oscillatory'
                b = 284.6;
                e = 1.5;
                a = 1.5;
                % c = c*b/d^e/sum(c);
                fun = @(x) cos(w(1)*2*pi+x*c');
            case 'productpeak'
                b = 725;
                e = 2;
                a = 5;
                % c = c*b/d^e/sum(c);
                fun = @(x) 1./prod(repmat(c.^-2,size(x,1),1) + ...
                    (x + repmat(w,size(x,1),1)).^2,2);
            case 'cornerpeak'
                b = 185;
                e = 2;
                a = 1.85;
                c = c*b/d^e/sum(c);
                fun = @(x) (1+x*c').^(-d-1);
            case 'gaussian'
                b = 70.3;
                e = 1;
                a = 7.03;
                % c = c*b/d^e/sum(c);
                % c(:)=1;
                % w(:)=0;
                fun = @(x) exp(-(x - repmat(w,size(x,1),1)).^2*(c'.^2));
            case  'continuous'
                b = 2040;
                e = 2;
                a = 20.4;
                % c = c*b/d^e/sum(c);
                fun = @(x) exp(-abs(x - repmat(w,size(x,1),1))*(c'.^2));
            case 'discontinuous'
                b = 430;
                e = 2;
                a = 4.3;
                % c = c*b/d^e/sum(c);
                fun = @(x) exp(x*c').*repmat(all(x<=repmat(w,size(x,1),1),2)',1,size(x,2));
        end
    case 'henonheiles'
        if nargin<=1
            d = 3;
        else
            d = varargin{1};
        end
        fun = @(x) 1/2*sum(x.^2,2) + ...
            0.2*sum(x(:,1:end-1).*x(:,2:end).^2-x(:,1:end-1).^3,2)+...
            0.2^2/16*sum((x(:,1:end-1).^2+x(:,2:end).^2).^2,2);
%         RV = RandomVector(UniformRandomVariable(-10,2),d);
        RV = RandomVector(NormalRandomVariable(),d);
        
    case 'sobol'
        % fun(x) = prod_{j=1}^d (|4*x_j-2|+a_j)/(1+a_j)
        if nargin<=1
            d = 8;
            a = [1,2,5,10,20,50,100,500];
        else
            d = varargin{1};
            if nargin <=2
                a = 2.^(0:d-1);
            else
                a = varargin{2};
            end
        end
        
        fun = @(x) prod((abs(4*x-2)+repmat(a,size(x,1),1))./repmat(1+a,size(x,1),1),2);
        RV = RandomVector(UniformRandomVariable(0,1),d);
    case 'anisotropic'
        % fun(x) = x_3*sin(x_4+x_16)
        if nargin<=1
            d = 16;
        else
            d = varargin{1};
        end
        fun = @(x) x(:,3).*sin(x(:,4)+x(:,16));
        RV = RandomVector(UniformRandomVariable(0,1),d);
    case 'polynomial'
        % fun(x) = 1/(2^d) * prod_{j=1}^d (3*(x_j)^q+1)
        if nargin<=1
            d = 16;
        else
            d = varargin{1};
            if nargin <=2
                q = 2;
            else
                q = varargin{2};
            end
        end
        fun = @(x) 1/(2^(size(x,2)))*prod(3*x.^q+1,2);
        RV = RandomVector(UniformRandomVariable(0,1),d);
    otherwise
        error('Bad FunctionName.')
end

rng('shuffle');