function [ mape,acc ] = eva( y_pd, y_true )

err=abs(y_pd-y_true)./y_true;
esn=0.3;
mape=sum(err)/length(y_true);
acc=sum(err<=esn)/length(y_true);


end


