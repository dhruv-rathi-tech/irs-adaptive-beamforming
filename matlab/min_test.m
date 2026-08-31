M = 5;
Atest = randn(M) + 1i*randn(M); Atest = (Atest+Atest')/2;
btest = randn(M,1) + 1i*randn(M,1);

cvx_begin sdp quiet
variable f_SDP
variable zeta(M)
maximize(f_SDP - sum(zeta))
subject to
zeta >= 0;
[Atest + diag(zeta), btest; ...
    btest', -f_SDP] >= 0;
cvx_end
disp('OK')