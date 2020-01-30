% Class MonteCarlo

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

classdef MonteCarlo
    
    properties
        N = 1e6 % Maximum number of Simulations
        std = 1e-2  % Desired Standard Deviation
        N0 = 100 % Number of Simulations for estimation of the Variance
    end
    
    methods
        
        function MC = MonteCarlo(N,std,N0)
            % MC = MonteCarlo(N,std,N0)
            % N : maximum number of simulations (default 10^6)
            % std : desired standard deviation (default 0.01)
            % N0 : number of simulations for variances estimation (default 100)
            
            if nargin >0
                MC.N = N;
                if nargin >1
                    MC.std = std;
                    if nargin >2
                        MC.N0 = N0;
                    end
                end
            end
        end
        
        function [I,output] = mean(MC,Y)
            % [I,output] = mean(MC,Y)
            % Estimate E(Y) using Monte-Carlo integration
            % MC.N: maximum number of samples
            % MC.std: desired standard deviation of Monte-Carlo estimate
            % MC.N0: number of samples used for estimating the variance
            %
            % Y: any object having methods size and random implemented
            % size(Y) returns an array sz
            % random(Y,N,1) must return N samples of Y in an array of size N-by-sz(1)-by-sz(2)
            
            sz = size(Y);
            if length(sz)==1
                sz = [sz,1];
            end
            
            assert(~(MC.N==Inf && MC.std==0),'Must have either N<Inf or std>0.')
            
            if MC.N0>MC.N
                MC.N0 = MC.N-1;
            end
            
            Ys = random(Y,MC.N0);
            vY = var(Ys,0,1);
            N = ceil(max(vY(:))/MC.std^2);
            
            if N>MC.N
                warning('precision can not be achieved with N=%d.',MC.N);
                N=MC.N;
            end
            
            Ysadd = random(Y,N-MC.N0);
            Ys = [Ys;Ysadd];
            
            I = reshape(mean(Ys,1),sz);
            vY = reshape(var(Ys,0,1),sz);
            output.std = sqrt(vY/N);
            output.N = N;
        end
        
        function [I,output] = meanOfSum(MC,U,Z)
            % [I,output] = meanOfSum(MC,U,Z)
            % Estimate mean of Y = U+Z, where U and Z are two real-valued random variables
            % E(Y) = E(U) + E(Z)
            %
            % U, Z: objects having method random. random(U,N) should
            % return a vector of N independent samples of U
            % MC.N: maximum number of samples of U
            % MC.std: desired standard deviation of Monte-Carlo estimate
            % MC.N0: number of samples used for estimating the variance
            %
            % First estimate the costs CZ and CU of sampling Z and U, and the variance VZ and VU of Z and U using MC.N0 samples of Z and U.
            % Then we determine the number of samples NZ and NU of Z and U in order to reach the desired precision with the minimal cost, or to minimize the variance for the maximal number of samples MC.N
            
            assert(~(MC.N==Inf && MC.std==0),'Must have either N<Inf or std>0.')
            
            CZ = clock();
            zs = random(Z,MC.N0);
            CZ = etime(clock(),CZ)/MC.N0;
            CU = clock();
            us = random(U,MC.N0);
            CU = etime(clock(),CU)/MC.N0;
            
            vZ = var(zs,0,1);
            vU = var(us,0,1);
            
            alpha = sqrt(vU./vZ*CZ/CU);
            
            NZ = ceil((vZ+vU./alpha)/MC.std^2);
            NU = ceil(alpha.*NZ);
            
            if any(NU>MC.N)
                NZ = ceil(MC.N./(1+alpha));
                NU = ceil(alpha.*NZ);
            end
            
            
            [NU,i] = max(NU(:)); % In case of vector-valued Y (to be checked)
            NZ = NZ(i);
            
            zsnew = random(Z,NZ-MC.N0);
            zs = [zs;zsnew];
            usnew = random(U,NU-MC.N0);
            us = [us;usnew];
            I = mean(zs,1) + mean(us,1);
            output.std = sqrt(var(zs,0,1)/NZ+var(us,0,1)/NU);
            output.CZ = CZ;
            output.CU = CU;
            output.NZ = NZ;
            output.NU = NU;
            output.VU = vU;
            output.VZ = vZ;
        end
        
        function [I,output] = meanControlVariateFunctionOfRandomVector(MC,Y,Z)
            % [I,output] = meanControlVariateFunctionOfRandomVector(MC,Y,Z)
            % Estimate mean of real-valued random variable Y = f(X) with control variate Z = g(X)
            % E(Y) = E(Z) + E(U) with U=Y-Z
            %
            % Y, Z: FunctionOfRandomVector
            % MC.N: maximum number of samples of Y
            % MC.std: desired standard deviation of Monte-Carlo estimate
            % MC.N0: number of samples used for estimating the variance
            %
            % First estimate the costs CZ, CY and CU of sampling Z, Y and U, and the variance VZ and VU of Z and U using MC.N0 samples of Z and Y.
            % Then we determine the number of samples NZ and NU of Z and U in order to reach the desired precision with the minimal cost, or to minimize the variance for the maximal number of samples MC.N
            
            U = FunctionOfRandomVector(UserDefinedFunction(@(x) Y.f.eval(x) - Z.f.eval(x),ndims(Y.X),size(Y.f)),Y.X);
            [I,output] = MC.meanOfSum(U,Z);
        end
    end
end