%% Gauss Integration Rule
% Gauss integration on [0,2]
mu = LebesgueMeasure(0, 2);
G = mu.gaussIntegrationRule(10);
% Function f(x) = exp(x)
f = UserDefinedFunction('exp(x1)', 1); 
I = G.integrate(f);
Iex = exp(2)-1;
fprintf('Error = %2.5e\n',(abs(I-Iex)/abs(Iex)))

%% Tensor product integration rule 
mu = ProductMeasure({LebesgueMeasure(0, 5), LebesgueMeasure(0, 1)});
G = mu.gaussIntegrationRule([4,2]);
%G = G.integrationRule();
% Function f(x1,x2) = exp(x1)*x2
f = UserDefinedFunction('exp(x1).*x2', 2);   
f.evaluationAtMultiplePoints = true; 
I = G.integrate(f);
Iex = (exp(5)-1)*1/2;
fprintf('Error = %2.5e\n', (abs(I-Iex)/abs(Iex)))
