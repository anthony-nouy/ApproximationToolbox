% Class ArchimedeanCopula

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

classdef ArchimedeanCopula
    
    properties
        d
        type
        a
        phi % Generator function
        iphi % Pseudo inverse of phi
        dphi
    end
    
    methods
        function C = ArchimedeanCopula(type,d,a)
            % C = ArchimedeanCopula(type,d,a)
            % Archimedean Copula in dimension d of type type, with parameter a
            % type = 'Clayton', a in [-1,infty)\{0}
            % type = 'Frank', a in R\{0}
            % type = 'Ali-Mikhail-Haq', a in [-1,1]
            %
            % type: char
            % a: double
            
            C.d = d;
            C.type = type;
            C.a=a;
            switch lower(type)
                
                case 'ali-mikhail-haq'
                    if a<-1 || a>1
                        error('Wrong value of parameter.')
                    end
                    C.phi = @(t) log((1-a*(1-t))./t);
                    C.iphi = @(t) (1-a)./(exp(t)-a);
                    C.dphi = @(t) (a-1)./t./(1-a*(1-t));
                case 'clayton'
                    if a<-1 || a==0
                        error('Wrong value of parameter.')
                    end
                    C.phi = @(t) 1/a*(t.^(-a)-1);
                    C.iphi = @(t) (1+a*t).^(-1/a);
                    C.dphi = @(t) -t.^(-a-1);
                case 'frank'
                    if a==0
                        error('Wrong value of parameter.')
                    end
                    C.phi = @(t) -log((exp(-a*t)-1)./(exp(-a)-1));
                    C.iphi = @(t)  -1/a*log(1+exp(-t)*(exp(-a)-1));
                    C.dphi = @(t) a./(1-exp(a*t));
                otherwise
                    error('Wrong type of Copula.')
            end
        end
        
        function c = cdf(C,u)
            c = C.iphi(sum(C.phi(u),2));
        end
        
        function p = pdf(C,u)
            p = dniphi(C,C.d,sum(C.phi(u),2)).*prod(C.dphi(u),2);
        end
    end
    
    methods (Hidden)
        function y = dniphi(C,n,t)
            switch lower(C.type)
                case 'ali-mikhail-haq'
                    b = zeros(n,n);
                    b(1,1)=-1;
                    for k=2:n
                        b(k,1)=b(k-1,1);
                        b(k,k)=-k*b(k-1,k-1);
                        b(k,2:k-1)=(2:k-1).*(b(k-1,2:k-1)-b(k-1,1:k-2));
                    end
                    y = zeros(length(t),1);
                    for i=1:n
                        y = y + b(n,i)*(exp(t)-C.a).^(-i).*exp(i*t);
                    end
                case 'clayton'
                    y = (-1)^n*prod(1+(1:n-1)*C.a)*(1+C.a*t).^(-1/C.a-n);
            end
        end
    end
end