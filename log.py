0
31
MatMul(dropout(relu(MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)+Var),Plh),Var,False,False)+Var
MatMul(dropout(relu(MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)+Var),Plh),Var,False,False)
dropout(relu(MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)+Var),Plh)
relu(MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)+Var)
MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)+Var
MatMul(reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136]),Var,False,False)
reshape(maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)),[-1, 3136])
maxpool(relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var))
relu(conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var)
conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)+Var
conv2d(maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)),Var)
maxpool(relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var))
relu(conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var)
conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)+Var
conv2d(reshape(Plh,[-1, 28, 28, 1]),Var)
reshape(Plh,[-1, 28, 28, 1])
15
{Var: None, Var: None, Var: None, Var: None, Var: None, Var: None, Const: array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float32), Plh: None, Plh: None, Var: None, Var: None, Const: array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float32), Const: array([0.1, 0.1, 0.1, ..., 0.1, 0.1, 0.1], dtype=float32), Const: array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
      dtype=float32), Plh: None}
[Plh, Plh, Var, Const, Var, Var, Const, Var, Var, Const, Var, Plh, Var, Const, Var]
