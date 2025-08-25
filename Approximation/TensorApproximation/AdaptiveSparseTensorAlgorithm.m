% Class AdaptiveSparseTensorAlgorithm

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

classdef AdaptiveSparseTensorAlgorithm
    
    properties
        addSamplesFactor
        nbSamples
        tol
        tolStagnation
        tolOverfit
        bulkParameter
        adaptiveSampling
        adaptationRule
        maxIndex
        display
        displayIterations
        fullOutput
    end
    
    methods
        
        function s = AdaptiveSparseTensorAlgorithm(varargin)
            % Class AdaptiveSparseTensorAlgorithm
            %
            % s = AdaptiveSparseTensorAlgorithm(varargin)
            % s.addSamplesFactor: percentage of additional samples, 0.1 by default
            % s.nbSamples: initial number of samples, 1 by default
            % s.tol: prescribed tolerance for cross-validation error, 1e-4 by default
            % s.tolStagnation: prescribed stagnation tolerance for cross-validation error, 5e-2 by default
            % s.tolOverfit: prescribed tolerance to detect overfitting for cross-validation error such that err>=toloverfit*err_old, 1.1 by default
            % s.bulkParameter: bulk parameter in (0,1), 0.5 by default
            % s.adaptiveSampling: adaptive sampling (true or false), true by default
            % s.adaptationRule: adaptation rule ('reducedMargin' or 'Margin'), 'reducedMargin' by default
            % s.maxIndex: maximal index, [] by default
            % s.display: display error and stagnation indicators at final step (true or false), false by default
            % s.displayIterations: display error and stagnation indicators at each step (true or false), false by default
            % s.fullOutput: convert sparse output to full output (true or false), false by default
            
            expectedAdaptationRules = {'reducedMargin','Margin'};
            
            p = ImprovedInputParser;
            addParamValue(p,'addSamplesFactor',0.1,@isscalar);
            addParamValue(p,'nbSamples',1,@isscalar);
            addParamValue(p,'tol',1e-4,@isscalar);
            addParamValue(p,'tolStagnation',5e-2,@isscalar);
            addParamValue(p,'tolOverfit',1.1,@isscalar);
            addParamValue(p,'bulkParameter',0.5,@isscalar);
            addParamValue(p,'adaptiveSampling',true,@islogical);
            addParamValue(p,'adaptationRule','reducedMargin',...
                @(x) any(validatestring(x,expectedAdaptationRules)));
            addParamValue(p,'maxIndex',[],@isscalar);
            addParamValue(p,'display',false,@islogical);
            addParamValue(p,'displayIterations',false,@islogical);
            addParamValue(p,'fullOutput',false,@islogical);
            
            parse(p,varargin{:});
            s = passMatchedArgsToProperties(p,s);
        end
        
        function [f,err,x,y,A] = leastSquares(s,fun,bases,ls,rv)
            % [f,err,x,y,A] = leastSquares(s,fun,bases,ls,rv)
            % Computes the least-squares approximation f of the function fun on
            % a tensor product of functional bases,
            % using sampling from the RandomVector rv if provided,
            % or from the standard RandomVector associated with each functional basis if not
            % s: AdaptiveSparseTensorAlgorithm
            % fun: Function
            % bases: FunctionalBases (should provide orthonormal bases) or
            % FullTensorProductFunctionalBasis
            % ls: LinearModelLearningSquareLoss
            % rv: RandomVector or RandomVariable (optional)
            % f: FunctionalBasisArray
            % err: 1-by-n doubles containing the (corrected) relative
            % (leave-one-out or k-fold) cross-validation error estimates
            % x: N-by-d doubles containing the evaluations of random variables
            % y: N-by-n doubles containing the evaluations of response vector
            % A: N-by-P doubles containing the evaluations of basis functions, where
            % N is the number of samples
            % d is the parametric dimension (number of random variables)
            % n is the output size of function fun
            % P is the basis dimension (number of basis functions)
            
            if ~isa(fun,'Function')
                error('Must provide a Function.')
            end
            
            if isa(bases,'FullTensorProductFunctionalBasis')
                bases = bases.bases;
            elseif isa(bases,'FunctionalBasis')
                bases = FunctionalBases({bases});
            end
            
            d = length(bases);
            I = MultiIndices(zeros(1,d));
            H = SparseTensorProductFunctionalBasis(bases,I);
            rvb = getRandomVector(bases);
            if nargin<5 || isempty(rv)
                rv = rvb;
            elseif isa(rv,'RandomVariable')
                rv = RandomVector(rv,d);
            end
            N = s.nbSamples;
            x = random(rv,N);
            fun = @(x) fun.eval(x);
            y = fun(x);
            if eq(rv,rvb)
                xb = x;
            else
                xb = transfer(rv,rvb,x);
            end
            A = H.eval(xb);
            
            if s.display || s.displayIterations
                fprintf('\n+-----------+------------+------------+\n');
                fprintf('| Dim basis | Nb samples |  CV error  |\n');
                fprintf('+-----------+------------+------------+\n');
            end
            f = [];
            err = Inf;
            basisAdaptation = true;
            ls.errorEstimation = true;
            while (norm(err) > s.tol) && basisAdaptation
                % Adaptive sampling on fixed basis
                [f,err,x,y,A] = s.adaptSampling(fun,H,ls,rv,f,err,x,y,A);
                % Adaptative basis with fixed sample
                [f,err,H,A,basisAdaptation] = s.adaptBasis(H,ls,rv,f,err,x,y,A,basisAdaptation);
            end
            
            if s.display && ~s.displayIterations
                fprintf('| %9d | %10d | %4.4e |\n',cardinal(H),size(x,1),norm(err));
            end
            
            if s.display || s.displayIterations
                fprintf('+-----------+------------+------------+\n');
            end
            
            if s.fullOutput
                f = fullOutputConversion(s,f);
            end
        end
        
        function [f,err,x,y,A] = leastSquaresCell(s,fun,bases,ls,rv)
            % [f,err,x,y,A] = leastSquaresCell(s,fun,bases,ls,rv,m)
            % Computes the least-squares approximation f of the function fun on
            % a tensor product of functional bases,
            % using sampling from the RandomVector rv if provided,
            % or from the standard RandomVector associated with each functional basis if not
            % s: AdaptiveSparseTensorAlgorithm
            % fun: CellValueduserDefinedFunction
            % bases: FunctionalBases (should provide orthonormal bases) or
            % FullTensorProductFunctionalBasis
            % ls: LinearModelLearningSquareLoss
            % rv: RandomVector or RandomVariable (optional)
            % f: cell array of size fun.outputSize containing objects of type FunctionalBasisArray
            % err: cell array of size fun.outputSize of 1-by-n_m doubles containing the (corrected) relative
            % (leave-one-out or k-fold) cross-validation error estimates
            % x: N-by-d doubles containing the evaluations of random variables
            % y: cell array of size fun.outputSize of N-by-n_m doubles containing the evaluations of response vector
            % A: cell array of size fun.outputSize of N-by-P_m doubles containing the evaluations of basis functions,
            % where
            % N is the number of samples
            % d is the parametric dimension (number of random variables)
            % n_m is the output size of the mth argument of function fun
            % P_m is the basis dimension (number of basis functions) for the mth argument of function fun
            
            if ~isa(fun,'Function')
                error('must provide a Function')
            end
            
            if isa(bases,'FullTensorProductFunctionalBasis')
                bases = bases.bases;
            elseif isa(bases,'FunctionalBasis')
                bases = FunctionalBases({bases});
            end
            
            d = length(bases);
            I = MultiIndices(zeros(1,d));
            basis = SparseTensorProductFunctionalBasis(bases,I);
            m = fun.outputSize;
            H = repmat({basis},m);
            rvb = getRandomVector(bases);
            if nargin<5 || isempty(rv)
                rv = rvb;
            elseif isa(rv,'RandomVariable')
                rv = RandomVector(rv,d);
            end
            N = s.nbSamples;
            x = random(rv,N);
            fun = @(x) fun.eval(x);
            y = fun(x);
            if eq(rv,rvb)
                xb = x;
            else
                xb = transfer(rv,rvb,x);
            end
            A = cellfun(@(basis) basis.eval(xb),H,'UniformOutput',false);
            
            if s.display || s.displayIterations
                fprintf('\n+-----------+------------+------------+\n');
                fprintf('| Dim basis | Nb samples |  CV error  |\n');
                fprintf('+-----------+------------+------------+\n');
            end
            f = cell(m);
            err = repmat({Inf},m);
            basisAdaptation = true(m);
            ls.errorEstimation = true;
            while any(cellfun(@(x) norm(x) > s.tol,err)) && any(basisAdaptation)
                % Adaptive sampling on fixed basis
                [f,err,x,y,A] = s.adaptSamplingCell(fun,H,ls,rv,f,err,x,y,A);
                % Adaptative basis with fixed sample
                parfor i=1:prod(m)
                    [f{i},err{i},H{i},A{i},basisAdaptation(i)] = s.adaptBasis(H{i},ls,rv,f{i},err{i},x,y{i},A{i},basisAdaptation(i));
                end
                if s.displayIterations && prod(m)~=1
                    fprintf( '|           |            |            |\n');
                end
            end
            
            if s.display && ~s.displayIterations
                for i=1:prod(m)
                    fprintf('| %9d | %10d | %4.4e |\n',cardinal(H{i}),size(x,1),norm(err{i}));
                end
            end
            
            if s.display || s.displayIterations
                fprintf('+-----------+------------+------------+\n');
            end
            
            if s.fullOutput
                f = cellfun(@(x) fullOutputConversion(s,x),f,'UniformOutput',false);
            end
        end
        
        function [f,output] = interpolate(s,fun,bases,grids,I)
            % [f,output] = interpolate(s,fun,bases,grids,I)
            % Computes a sparse tensor interpolation of the function f
            % on a tensor product of orthonormal basis
            % s: AdaptiveSparseTensorInterpolation
            % fun: Function
            % bases: FunctionalBases (should provide orthonormal bases) or
            % FullTensorProductFunctionalBasis
            % grids: cell containing interpolation grids for each dimension
            % if grids = [], use magic grids
            % I: MultiIndices, initial multi-index set
            % f: FunctionalBasisArray
            
            if ~isa(fun,'Function')
                error('must provide a Function')
            end
            
            if isa(bases,'FullTensorProductFunctionalBasis')
                bases = bases.bases;
            elseif isa(bases,'FunctionalBasis')
                bases = FunctionalBases({bases});
            end
            
            d = length(bases);
            nd = cellfun(@ndims,bases.bases);
            
            if isempty(s.maxIndex)
                s.maxIndex = reshape((cardinals(bases))-1,1,d);
            elseif length(s.maxIndex)==1
                s.maxIndex = repmat(s.maxIndex,1,d);
            end
            
            if nargin<5
                I = MultiIndices(zeros(1,d));
            end
            
            if nargin<4 ||  isempty(grids)
                rvb = ProductMeasure(getRandomVector(bases));
                grids = mat2cell(random(rvb,1000),1000,nd);
            end
            
            if isa(grids,'FullTensorGrid')
                grids = grids.grids;
            elseif ~isa(grids,'cell')
                grids = {grids};
            end
            
            for k=1:length(bases)
                grids{k} = magicPoints(bases.bases{k},grids{k},1:cardinal(bases.bases{k}));
            end
            
            err = Inf;
            
            numberOfEvaluations = 0;
            fun.store = true;
            H = SparseTensorProductFunctionalBasis(bases,I);
            x = SparseTensorGrid(grids,I+1);
            x = array(x);
            
            [y,fun] = fun(x);
            
            f = H.interpolate(y,x);
            
            numberOfEvaluations = numberOfEvaluations + size(x,1);
            
            while (err > s.tol)
                switch lower(s.adaptationRule)
                    case 'margin'
                        Iadd = getMargin(I);
                    case 'reducedmargin'
                        Iadd = getReducedMargin(I);
                end
                rem = find((~(Iadd<=MultiIndices(s.maxIndex))));
                Iadd = removeIndices(Iadd,rem);
                if cardinal(Iadd)==0
                    break
                end
                
                Itest = addIndices(I,Iadd);
                Htest = SparseTensorProductFunctionalBasis(bases,Itest);
                xadd = SparseTensorGrid(grids,Iadd+1);
                xadd = array(xadd);
                [yadd,fun] = fun(xadd);
                numberOfEvaluations = numberOfEvaluations + size(xadd,1);
                ytest = [y;yadd];
                xtest = [x;xadd];
                ftest = Htest.interpolate(ytest,xtest);
                [~,loc] = ismember(Iadd.array,Itest.array,'rows');
                a = ftest.data;
                a_marg = a(loc,:);
                norm_a_marg = sqrt(sum(a_marg.^2,2));
                err = norm(a_marg,'fro')/norm(a,'fro');
                
                if s.displayIterations
                    fprintf('Dimension = %d, Error = %d\n',cardinal(H),err);
                end
                
                switch lower(s.adaptationRule)
                    case 'margin'
                        env = envelope(Iadd,norm_a_marg);
                        [~,ind] = sort(env,'descend');
                    case 'reducedmargin'
                        [~,ind] = sort(norm_a_marg,'descend');
                end
                
                energy = cumsum(norm_a_marg(ind).^2);
                rep = find(energy>=s.bulkParameter.*energy(end)) ;
                Iadd.array = Iadd.array(ind(1:rep(1)),:);
                I = I.addIndices(Iadd);
                xadd = xadd(ind(1:rep(1)),:);
                yadd = yadd(ind(1:rep(1)),:);
                y = [y;yadd];
                x = [x;xadd];
                H = SparseTensorProductFunctionalBasis(bases,I);
                f = H.interpolate(y,x);
            end
            output.error = err;
            output.grids = grids;
            output.grid = x;
            
            if fun.store
                output.numberOfEvaluations = size(fun.xStored,1);
            else
                output.numberOfEvaluations = numberOfEvaluations;
            end
            
            if s.fullOutput
                f = fullOutputConversion(s,f);
            end
        end
        
        
        function [f,output] = interpolatePerturbed(s,fun,bases,grids,I)
            % [f,output] = interpolatePerturbed(s,fun,bases,grids,I)
            % Computes a sparse tensor interpolation from inexact evaluation of
            % the function f on a tensor product of orthonormal basis
            % s: AdaptiveSparseTensorInterpolation
            % fun:function with two arguments
            % - x:
            % - H: SparseTensorProductFunctionalBasis
            % bases: FunctionalBases (should provide orthonormal bases)
            % grids: cell containing interpolation grids for each dimension
            % if grids = [], use magic grids
            % I: MultiIndices, initial multi-index set
            % f: FunctionalBasisArray
            
            d = length(bases);
            
            if isempty(s.maxIndex)
                s.maxIndex = reshape((cardinals(bases))-1,1,d);
            elseif length(s.maxIndex)==1
                s.maxIndex = repmat(s.maxIndex,1,d);
            end
            
            if nargin<5
                I = MultiIndices(zeros(1,d));
            end
            
            if nargin<4 ||  isempty(grids)
                rvb = getRandomVector(bases);
                grids = mat2cell(random(rvb,1000),1000,ones(1,d));
            end
            
            for k=1:length(bases)
                grids{k} = magicPoints(bases.bases{k},grids{k},1:cardinal(bases.bases{k}));
            end
            
            err = Inf;
            
            numberOfEvaluations = 0;
            
            H = SparseTensorProductFunctionalBasis(bases,I);
            x = SparseTensorGrid(grids,I+1);
            x = array(x);
            [~,f]   = fun(x,H);
            
            numberOfEvaluations = numberOfEvaluations + size(x,1);
            
            while (err > s.tol)
                switch lower(s.adaptationRule)
                    case 'margin'
                        Iadd = getMargin(I);
                    case 'reducedmargin'
                        Iadd = getReducedMargin(I);
                end
                rem = find((~(Iadd<=MultiIndices(s.maxIndex))));
                Iadd = removeIndices(Iadd,rem);
                
                if cardinal(Iadd)==0
                    break
                end
                
                Itest   = addIndices(I,Iadd);
                Htest   = SparseTensorProductFunctionalBasis(bases,Itest);
                xadd    = SparseTensorGrid(grids,Iadd+1);
                xadd    = array(xadd);
                
                numberOfEvaluations = numberOfEvaluations + size(xadd,1);
                
                xtest     = [x;xadd];
                [~,ftest] = fun(xtest,Htest);
                [~,loc]   = ismember(Iadd.array,Itest.array,'rows');
                a         = ftest.data;
                a_marg    = a(loc,:);
                norm_a_marg = sqrt(sum(a_marg.^2,2));
                
                err = norm(a_marg,'fro')/norm(a,'fro');
                
                if s.displayIterations
                    fprintf('Dimension = %d, Error = %d\n',cardinal(H),err);
                end
                
                switch lower(s.adaptationRule)
                    case 'margin'
                        env = envelope(Iadd,norm_a_marg);
                        [~,ind] = sort(env,'descend');
                    case 'reducedmargin'
                        [~,ind] = sort(norm_a_marg,'descend');
                end
                
                energy = cumsum(norm_a_marg(ind).^2);
                rep = find(energy>=s.bulkParameter.*energy(end)) ;
                Iadd.array = Iadd.array(ind(1:rep(1)),:);
                I = I.addIndices(Iadd);
                
                xadd = xadd(ind(1:rep(1)),:);
                x    = [x;xadd];
                
                H = SparseTensorProductFunctionalBasis(bases,I);
                [~,f] = fun(x,H);
            end
            
            output.error = err;
            output.grids = grids;
            output.grid = x;
            
            output.numberOfEvaluations = numberOfEvaluations;
            
            if s.fullOutput
                f = fullOutputConversion(s,f);
            end
        end
    end
    
    methods (Hidden)
        function f = fullOutputConversion(~,f)
            if  length(cardinals(f.basis.bases)) == 1
                bases = f.basis.bases;
                I = f.basis.indices +1;
                basis = FullTensorProductFunctionalBasis(bases);
                data = sparse(cardinal(basis),size(f.data,2));
                data(I.array) = f.data;
                f.basis = basis;
                f.data = full(data);
            else
                bases = f.basis.bases;
                I = f.basis.indices +1;
                basis = FullTensorProductFunctionalBasis(bases);
                ind = sub2ind(I,cardinals(bases)');
                data = sparse(cardinal(basis),size(f.data,2));
                data(ind,:) = f.data;
                f.basis = basis;
                f.data = full(data);
            end
        end
        
        function [f,err,H,A,basisAdaptation] = adaptBasis(s,H,ls,rv,f,err,x,y,A,basisAdaptation)
            % [f,err,H,A,basisAdaptation] = adaptBasis(s,H,ls,rv,f,err,x,y,A,basisAdaptation)
            
            if ~basisAdaptation
                return
            end
            
            N = size(x,1);
            bases = H.bases;
            I = H.indices;
            
            d = size(I.array,2);
            if isempty(s.maxIndex)
                s.maxIndex = reshape(cardinals(bases)-1,1,d);
            elseif length(s.maxIndex)==1
                s.maxIndex = repmat(s.maxIndex,1,d);
            end
            
            rvb = getRandomVector(H);
            basisAdaptation = true;
            % err_stagn = Inf;
            sz = size(y);
            yls = reshape(y,[sz(1) prod(sz(2:end))]);
            while (norm(err) > s.tol)% && (err_stagn > s.tolStagnation)
                switch lower(s.adaptationRule)
                    case 'margin'
                        Iadd = getMargin(I);
                    case 'reducedmargin'
                        Iadd = getReducedMargin(I);
                end
                
                rem = find((~(Iadd<=MultiIndices(s.maxIndex))));
                Iadd = removeIndices(Iadd,rem);
                
                if cardinal(Iadd)==0
                    basisAdaptation = false;
                    break
                end
                
                Itest = I.addIndices(Iadd);
                if cardinal(Itest) > N% && ~ls.regularization
                    break
                end
                Htest = SparseTensorProductFunctionalBasis(bases,Itest);
                if eq(rv,rvb)
                    xb = x;
                else
                    xb = transfer(rv,rvb,x);
                end
                ls.basis = [];
                ls.basisEval = Htest.eval(xb);
                ls.trainingData = {[], yls};
                [a,~] = ls.solve();
                [~,loc] = ismember(Iadd.array,Itest.array,'rows');
                a_marg = a(loc,:);
                norm_a_marg = sqrt(sum(a_marg.^2,2));
                switch lower(s.adaptationRule)
                    case 'margin'
                        env = envelope(Iadd,norm_a_marg);
                        [~,ind] = sort(env,'descend');
                    case 'reducedmargin'
                        [~,ind] = sort(norm_a_marg,'descend');
                end
                energy = cumsum(norm_a_marg(ind).^2);
                rep = find(energy>=s.bulkParameter.*energy(end)) ;
                Iadd.array = Iadd.array(ind(1:rep(1)),:);
                I = I.addIndices(Iadd);
                H = SparseTensorProductFunctionalBasis(bases,I);
                A = H.eval(xb);
                ls.basis = [];
                ls.basisEval = A;
                ls.trainingData = {[], yls};
                [a,output] = ls.solve();
                a = reshape(a,[size(a,1) sz(2:end)]);
                f = FunctionalBasisArray(a,H,sz(2:end));
                err_stagn = norm(output.error-err)/norm(output.error);
                err_old = err;
                err = output.error;
                if s.displayIterations
                    fprintf('| %9d |            | %4.4e |\n',cardinal(H),norm(err));
                end
                if (norm(err) > s.tolOverfit*norm(err_old)) || (err_stagn <= s.tolStagnation && ls.regularization)
                    break
                end
            end
        end
        
        function [f,err,x,y,A] = adaptSampling(s,fun,H,ls,rv,f,err,x,y,A)
            % [f,err,x,y,A] = adaptSampling(s,fun,H,ls,rv,f,err,x,y,A)
            
            N = size(x,1);
            rvb = getRandomVector(H);
            err_stagn = Inf;
            while (norm(err) > s.tol) && (err_stagn > s.tolStagnation) && s.adaptiveSampling
                Nadd = ceil(s.addSamplesFactor*N);
                N = N + Nadd;
                xadd = random(rv,Nadd);
                x = [x;xadd];
                yadd = fun(xadd);
                y = [y;yadd];
                if eq(rv,rvb)
                    xbadd = xadd;
                else
                    xbadd = transfer(rv,rvb,xadd);
                end
                Aadd = H.eval(xbadd);
                A = [A;Aadd];
                ls.basis = [];
                ls.basisEval = A;
                sz = size(y);
                ls.trainingData = {[], reshape(y,[sz(1) prod(sz(2:end))])};
                [a,output] = ls.solve();
                a = reshape(a,[size(a,1) sz(2:end)]);
                f = FunctionalBasisArray(a,H,sz(2:end));
                err_stagn = norm(output.error-err)/norm(output.error);
                err = output.error;
                if s.displayIterations
                    fprintf('|           | %10d | %4.4e |\n',N,norm(err));
                end
            end
        end
        
        function [f,err,x,y,A] = adaptSamplingCell(s,fun,H,ls,rv,f,err,x,y,A)
            % [f,err,x,y,A] = adaptSamplingCell(s,fun,H,ls,rv,f,err,x,y,A)
            
            m = numel(y);
            N = size(x,1);
            rvb = cellfun(@(basis) getRandomVector(basis),H,'UniformOutput',false);
            err_stagn = Inf;
            while any(cellfun(@(x) norm(x) > s.tol,err)) && any(err_stagn > s.tolStagnation) && s.adaptiveSampling
                Nadd = ceil(s.addSamplesFactor*N);
                N = N + Nadd;
                xadd = random(rv,Nadd);
                x = [x;xadd];
                yadd = fun(xadd);
                y = cellfun(@(x,xadd) [x;xadd],y,yadd,'UniformOutput',false);
                xbadd = cell(m,1);
                for i=1:m
                    if eq(rv,rvb{i})
                        xbadd{i} = xadd;
                    else
                        xbadd{i} = transfer(rv,rvb{i},xadd);
                    end
                end
                Aadd = cellfun(@(basis,x) basis.eval(x),H,xbadd,'UniformOutput',false);
                A = cellfun(@(x,xadd) [x;xadd],A,Aadd,'UniformOutput',false);
                ls.basis = [];
                parfor i=1:m
                    sz = size(y{i});
                    lsloc = ls;
                    lsloc.trainingData = {[], reshape(y{i},[sz(1) prod(sz(2:end))])};
                    lsloc.basisEval = A{i};
                    [a,output] = lsloc.solve();
                    a = reshape(a,[size(a,1) sz(2:end)]);
                    f{i} = FunctionalBasisArray(a,H{i},sz(2:end));
                    err_stagn(i) = norm(output.error-err{i})/norm(output.error);
                    err{i} = output.error;
                    
                    if s.displayIterations
                        fprintf('|           | %10d | %4.4e |\n',N,norm(err{i}));
                    end
                end
                if s.displayIterations && m~=1
                    fprintf('|           |            |            |\n');
                end
            end
        end
    end
end