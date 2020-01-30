% PLOTMULTIINDICES - Plot the multi-index set
%
% PLOTMULTIINDICES(I,'dim',dim)
% Plot the multi-index set projected on dimension dim
% dim = 1:min(3,ndims(I)) by default
% 
% PLOTMULTIINDICES(I,'dim',dim,'MarkerArea',MarkerArea,'MarkerColor',MarkerColor,'MarkerType',MarkerType,'axis',axis)
% Plot the multi-index set projected on dimension dim with optional parameters
% marker = 'o' by default
% color = 'b' by default
%
% PLOTMULTIINDICES(I,'dim',dim,'maximal',1)
% Plot the multi-index set projected on dimension dim and its maximal indices
%
% PLOTMULTIINDICES(I,'dim',dim,'margin',1)
% Plot the multi-index set projected on dimension dim and its margin
%
% PLOTMULTIINDICES(I,'dim',dim,'reducedmargin',1)
% Plot the multi-index set projected on dimension dim and its reduced margin

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

function plotMultiIndices(I,varargin)
if isa(I,'double')
    I = MultiIndices(I);
end
d = ndims(I);

isboolean = @(x) islogical(x) || isnumeric(x) && all(x(:)==0 | x(:)==1);

p = inputParser;
addParamValue(p,'dim',1:min(3,d),@isnumeric);
addParamValue(p,'MarkerArea',36,@isnumeric);
addParamValue(p,'MarkerColor','b',@(x) isnumeric(x) || ischar(x));
addParamValue(p,'MarkerType','o',@ischar);
addParamValue(p,'MarkerArea_max',216,@isnumeric);
addParamValue(p,'MarkerType_max','s',@ischar);
addParamValue(p,'MarkerColor_marg','r',@(x) isnumeric(x) || ischar(x));
addParamValue(p,'legend',true,isboolean);
addParamValue(p,'label',true,isboolean);
addParamValue(p,'grid',true,isboolean);
addParamValue(p,'axis',[],@isnumeric)
addParamValue(p,'tick',[],@(x) isnumeric(x) || iscell(x))
addParamValue(p,'maximal',false,isboolean);
addParamValue(p,'margin',false,isboolean);
addParamValue(p,'reducedMargin',false,isboolean);
addParamValue(p,'FontSize',16,@isscalar);
addParamValue(p,'LineWidth',1,@isscalar);
addParamValue(p,'Interpreter','latex',@ischar);
parse(p,varargin{:})

Idim = I;
Idim.array = unique(I.array(:,p.Results.dim),'rows');

switch length(p.Results.dim)
    case 0
        error('The number of dimensions must be greater or equal to 1.')
    case 1
        scatter(Idim.array(:,1),zeros(size(Idim.array,1),1),p.Results.MarkerArea,p.Results.MarkerColor,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor,'MarkerFaceColor',p.Results.MarkerColor)
        axis equal
        set(gca,'FontSize',p.Results.FontSize)
        if ~isempty(p.Results.tick)
            set(gca,'XTick',p.Results.tick,'YTick',[])
        else
            set(gca,'XTick',0:max(Idim.array(:,1)),'YTick',[])
        end
        if ~isempty(p.Results.axis)
            axis(p.Results.axis)
        else
            set(gca,'XLim',[-eps,max(Idim.array(:,1))+1],'YLim',[-eps,1])
        end
        leg = {'$\alpha \in \Lambda$'};
        hold on
        
        if p.Results.maximal
            Idim_max = getMaximalIndices(Idim);
            scatter(Idim_max.array(:,1),zeros(size(Idim_max.array,1),1),p.Results.MarkerArea_max,p.Results.MarkerColor,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
            leg = [leg, {'$\max(\Lambda)$'}];
        end
        
        if p.Results.margin
            Idim_marg = getMargin(Idim);
            scatter(Idim_marg.array(:,1),zeros(size(Idim_marg.array,1),1),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick,'YTick',[])
            else
                set(gca,'XTick',0:max(Idim_marg.array(:,1)),'YTick',[])
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Idim_marg.array(:,1))+1],'YLim',[-eps,1])
            end
            leg = [leg, {'$\mathcal{M}(\Lambda)$'}];
            if p.Results.reducedMargin
                Idim_red = getReducedMargin(Idim);
                scatter(Idim_red.array(:,1),zeros(size(Idim_red.array,1),1),p.Results.MarkerArea_max,p.Results.MarkerColor_marg,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
                leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
            end
        elseif p.Results.reducedMargin
            Idim_red = getReducedMargin(Idim);
            scatter(Idim_red.array(:,1),zeros(size(Idim_red.array,1),1),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick,'YTick',[])
            else
                set(gca,'XTick',0:max(Idim_red.array(:,1)),'YTick',[])
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Idim_red.array(:,1))+1],'YLim',[-eps,1])
            end
            leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
        end
        if p.Results.label
            xlabel(['$\alpha_{' num2str(p.Results.dim) '}$'],'Interpreter',p.Results.Interpreter)
        end
        hold off
    case 2
        scatter(Idim.array(:,1),Idim.array(:,2),p.Results.MarkerArea,p.Results.MarkerColor,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor,'MarkerFaceColor',p.Results.MarkerColor)
        axis equal
        set(gca,'FontSize',p.Results.FontSize)
        if ~isempty(p.Results.tick)
            set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2})
        else
            set(gca,'XTick',0:max(Idim.array(:,1)),'YTick',0:max(Idim.array(:,2)))
        end
        if ~isempty(p.Results.axis)
            axis(p.Results.axis)
        else
            set(gca,'XLim',[-eps,max(Idim.array(:,1))+1],'YLim',[-eps,max(Idim.array(:,2))+1])
        end
        leg = {'$\Lambda$'};
        hold on
        if p.Results.maximal
            Imax = getMaximalIndices(Idim);
            scatter(Imax.array(:,1),Imax.array(:,2),p.Results.MarkerArea_max,p.Results.MarkerColor,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
            leg = [leg, {'$\max(\Lambda)$'}];
        end
        if p.Results.margin
            Imarg = getMargin(Idim);
            scatter(Imarg.array(:,1),Imarg.array(:,2),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2})
            else
                set(gca,'XTick',0:max(Imarg.array(:,1)),'YTick',0:max(Imarg.array(:,2)))
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Imarg.array(:,1))+1],'YLim',[-eps,max(Imarg.array(:,2))+1])
            end
            leg = [leg, {'$\mathcal{M}(\Lambda)$'}];
            if p.Results.reducedMargin
                Imarg_red = getindices(Idim,'margin_reduced');
                scatter(Imarg_red.array(:,1),Imarg_red.array(:,2),p.Results.MarkerArea_max,p.Results.MarkerColor_marg,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
                leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
            end
        elseif p.Results.reducedMargin
            Idim_red = getReducedMargin(Idim);
            scatter(Idim_red.array(:,1),Idim_red.array(:,2),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2})
            else
                set(gca,'XTick',0:max(Idim_red.array(:,1)),'YTick',0:max(Idim_red.array(:,2)))
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Idim_red.array(:,1))+1],'YLim',[-eps,max(Idim_red.array(:,2))+1])
            end
            leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
        end
        if p.Results.label
            xlabel(['$\alpha_{' num2str(p.Results.dim(1)) '}$'],'Interpreter',p.Results.Interpreter)
            ylabel(['$\alpha_{' num2str(p.Results.dim(2)) '}$'],'Interpreter',p.Results.Interpreter)
        end
        hold off
    case 3
        scatter3(Idim.array(:,1),Idim.array(:,2),Idim.array(:,3),p.Results.MarkerArea,p.Results.MarkerColor,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor,'MarkerFaceColor',p.Results.MarkerColor)
        axis equal
        % view(37.5,30)
        view(50,10)
        set(gca,'FontSize',p.Results.FontSize)
        if ~isempty(p.Results.tick)
            set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2},'ZTick',p.Results.tick{3})
        else
            set(gca,'XTick',0:max(Idim.array(:,1)),'YTick',0:max(Idim.array(:,2)),'ZTick',0:max(Idim.array(:,3)))
        end
        if ~isempty(p.Results.axis)
            axis(p.Results.axis)
        else
            set(gca,'XLim',[-eps,max(Idim.array(:,1))+1],'YLim',[-eps,max(Idim.array(:,2))+1],'ZLim',[-eps,max(Idim.array(:,3))+1])
        end
        leg = {'$\Lambda$'};
        hold on
        if p.Results.maximal
            Imax = getMaximalIndices(Idim);
            scatter3(Imax.array(:,1),Imax.array(:,2),Imax.array(:,3),p.Results.MarkerArea*4,p.Results.MarkerColor,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
            leg = [leg, {'$\max(\Lambda)$'}];
        end
        if p.Results.margin
            Imarg = getMargin(Idim);
            scatter3(Imarg.array(:,1),Imarg.array(:,2),Imarg.array(:,3),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2},'ZTick',p.Results.tick{3})
            else
                set(gca,'XTick',0:max(Imarg.array(:,1)),'YTick',0:max(Imarg.array(:,2)),'ZTick',0:max(Imarg.array(:,3)))
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Imarg.array(:,1))+1],'YLim',[-eps,max(Imarg.array(:,2))+1],'ZLim',[-eps,max(Imarg.array(:,3))+1])
            end
            leg = [leg, {'$\mathcal{M}(\Lambda)$'}];
            if p.Results.reducedMargin
                Idim_red = getReducedMargin(Idim);
                scatter3(Idim_red.array(:,1),Idim_red.array(:,2),Idim_red.array(:,3),p.Results.MarkerArea_max,p.Results.MarkerColor_marg,p.Results.MarkerType_max,'LineWidth',p.Results.LineWidth)
                leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
            end
        elseif p.Results.reducedMargin
            Idim_red = getReducedMargin(Idim);
            scatter3(Idim_red.array(:,1),Idim_red.array(:,2),Idim_red.array(:,3),p.Results.MarkerArea,p.Results.MarkerColor_marg,p.Results.MarkerType,'LineWidth',p.Results.LineWidth,'MarkerEdgeColor',p.Results.MarkerColor_marg,'MarkerFaceColor',p.Results.MarkerColor_marg)
            if ~isempty(p.Results.tick)
                set(gca,'XTick',p.Results.tick{1},'YTick',p.Results.tick{2},'ZTick',p.Results.tick{3})
            else
                set(gca,'XTick',0:max(Idim_red.array(:,1)),'YTick',0:max(Idim_red.array(:,2)),'ZTick',0:max(Idim_red.array(:,3)))
            end
            if ~isempty(p.Results.axis)
                axis(p.Results.axis)
            else
                set(gca,'XLim',[-eps,max(Idim_red.array(:,1))+1],'YLim',[-eps,max(Idim_red.array(:,2))+1],'ZLim',[-eps,max(Idim_red.array(:,3))+1])
            end
            leg = [leg, {'$\mathcal{M}_r(\Lambda)$'}];
        end
        if p.Results.label
            xlabel(['$\alpha_{' num2str(p.Results.dim(1)) '}$'],'Interpreter',p.Results.Interpreter)
            ylabel(['$\alpha_{' num2str(p.Results.dim(2)) '}$'],'Interpreter',p.Results.Interpreter)
            zlabel(['$\alpha_{' num2str(p.Results.dim(3)) '}$'],'Interpreter',p.Results.Interpreter)
        end
        hold off
    otherwise
        error('The number of dimensions must not exceed 3 to display.')
end
if p.Results.grid
    grid on
end
if p.Results.legend
    l = legend(leg{:});
    set(l,'Interpreter',p.Results.Interpreter)
end
end
