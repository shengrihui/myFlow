syms x k b y linear
linear = k * x + b;
sigmoid = 1./(1+exp(-linear));
loss = (sigmoid-y).^2
dk=diff(loss,k)
db=diff(loss,b)

x=30
y=10
k=123
b=110.38
dbb=subs(db)



