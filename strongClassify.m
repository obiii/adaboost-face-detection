function [h_sx,cM,acc] = strongClassify(nbrWeakClassifiers,alpha,polarity,features,threshold,X,y)

X = X;
result = 0;
for i = 1: nbrWeakClassifiers
   t = threshold(i,1);
   p = polarity(i,1);
   f = features(i,1);
   h_x = WeakClassifier(t,p,X(f,:));
   a_h_x = alpha(i,1) .* h_x;
   
   result = result+a_h_x;
end

h_sx = sign(result);
cM = confusionmat(h_sx,y);
acc = sum(diag(cM))/sum(cM,'all');

end