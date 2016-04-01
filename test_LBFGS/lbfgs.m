% initialize L-BFGS
memory=10;
S  = zeros(n, memory);
Y  = zeros(n, memory);
YS = zeros(memory, 1);
LBFGS_col = 1;LBFGS_mem = 0;skipCount = 0;

% calculate Tx (here known as Rx)

d = - Rx;% stepsize should be always equal to one but acceleration might result through K0
if iter == 1
    d = - Rx;
    LBFGS_col = 1;
    LBFGS_mem = 0;
else
    Sk  = x - xold;
    Yk  = Rx - Rxold;
    YSk = Yk'*Sk;
    if nrmRx < 1,alphaC = 3;end
    if YSk/(Sk'*Sk) > 1e-6*nrmRx^alphaC
        LBFGS_col = 1 + mod(LBFGS_col, memory);
        LBFGS_mem = min(LBFGS_mem+1, memory);
        S(:,LBFGS_col) = Sk;
        Y(:,LBFGS_col) = Yk;
        YS(LBFGS_col)  = YSk;
    else
        skipCount = skipCount+1;
    end
    H = YSk/(Yk'*Yk);
    d = LBFGS(S, Y, YS, H, -Rx, int32(LBFGS_col), int32(LBFGS_mem));
end

% calculate the new x based on a line search, say Armijo's
% conditions