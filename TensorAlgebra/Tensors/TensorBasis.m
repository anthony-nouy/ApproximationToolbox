% Class TensorBasis: algebraic tensors in tree-based tensor format

% Copyright (c) 2020, Anthony Nouy, Erwan Grelier, Loic Giraldi
%
% This file is part of ApproximationToolbox.
%

classdef TensorBasis

    properties
        % TENSOR - Tensor representing a basis of tensors
        tensor = []
        % ORDER - order of the tensor space
        order = []
        % ISORTH - True if the basis of orthonormal
        isOrth = false
        % SZ
        sz
    end

    methods
        function x = TensorBasis(t)
            % TensorBasis - Constructor for the class TensorBasis
            %
            % x = TensorBasis(t)
            % t: AlgebraicTensor
            % x: TensorBasis
            if nargin>0
                x.tensor = t;

                if isa(t,'TreeBasedTensor')
                    x.order = t.order;
                    x.sz = t.sz;
                    a = t.tensors{t.tree.root};
                    nbCh = length(nonzeros(t.tree.children(:,t.tree.root)));
                    if a.order~=nbCh+1
                        error('the root tensor should have order equal to number of children + 1')
                    end
                elseif isa(t,'FullTensor')
                    x.order = t.order - 1;
                    x.sz = t.sz(1:x.order);
                else
                    error('Constructor not implemented.');
                end
            end
        end

        function BI = evalAtIndices(B,I)

            x = B.tensor;
            if isa(x,'TreeBasedTensor')
                BI = x.evalAtIndices(I);
            elseif isa(x,'FullTensor')
                BI = x.evalAtIndices(I,1:x.order-1);
            end


        end

        function n = cardinal(B)
            x = B.tensor;
            if isa(x,'TreeBasedTensor')
                a = x.tensors{x.tree.root};
                n = size(a,a.order);
            elseif isa(x,'FullTensor')
                n = size(x,x.order);
            end
        end

        function cd = christoffel(B,I)

            if nargin==2
                BI = double(B.evalAtIndices(I));
                cd = sum(BI.^2,2)/size(BI,2);
            else

                x = B.tensor;

                if isa(x,'FullTensor')
                    cd = sum(x.*x,x.order)/x.sz(x.order);
                    cd = squeeze(cd,x.order);
                elseif isa(x,'TreeBasedTensor')

                    cd = x.*x;
                    a = cd.tensors{cd.tree.root};
                    b = sum(a,a.order)/a.sz(a.order);
                    if isa(b,'double')
                        b = FullTensor(b,a.order,[a.sz(1:a.order-1),1]);
                    end
                    a = squeeze(b,a.order);



                    cd.tensors{cd.tree.root} = a;
                    cd = cd.updateProperties();
                else
                    error('wrong type')
                end
            end

        end


        function I = iidOptimalSampling(B,N)


            if isa(B.tensor,'FullTensor')
                cd = B.christoffel();
                I = sampleFromTensorProbabilityTable(cd,N);

            elseif isa(B.tensor,'TreeBasedTensor')
                error('not implemented')
            else
                error('wrong type')
            end

        end


        function I = dppSampling(B,L)
            if nargin==1
                L=1;
            end

            if isa(B.tensor,'FullTensor')
                m = cardinal(B);
                I = cell(L,1);
                I(:) = {zeros(m,B.tensor.order-1)};
                
                parfor l=1:L                    

                    for k=1:m

                        if k>1
                            p = timesMatrix(B.tensor,eye(m) - V*V',B.tensor.order);
                        else
                            V = zeros(m,0);
                            p = B.tensor;
                        end
                        p = sum(p.*p,p.order)/(p.sz(p.order)-k+1);
                        p = squeeze(p,p.order);

                        newI = sampleFromTensorProbabilityTable(p,1);
                        V(:,end+1) = B.evalAtIndices(newI).';
                        V = orth(V);
                        I{l}(k,:) = newI;
                    end

                end

                I = vertcat(I{:});


            elseif isa(B.tensor,'TreeBasedTensor')

                m = cardinal(B);
                I = zeros(m*L,B.tensor.order);
                t = B.tensor.tree;
                

                for l=1:L

                    for k=1:m

                        if k>1
                            p = timesMatrix(B.tensor.tensors{t.root},eye(m) - V*V',B.tensor.tensors{t.root}.order);
                            B.tensor.tensors{t.root}=p;
                            p = B.tensor;
                        else
                            V = zeros(m,0);
                            p = B.tensor;
                        end
                        p = full(p);
                        p = sum(p.*p,p.order)/(p.sz(p.order)-k+1);
                        p = squeeze(p,p.order);

                        newI = sampleFromTensorProbabilityTable(p,1);
                        V(:,end+1) = B.evalAtIndices(newI);
                        V = orth(V);
                        I(k+(l-1)*m,:) = newI;
                    end


                end
                                

            else
                error('wrong type')
            end


        end

    end


    methods (Static)



    end
end