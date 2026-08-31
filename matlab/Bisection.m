function  m = Bisection(f, low, high, tol)
%disp('Bisection Method');
i = 0;
% Evaluate both ends of the interval
y1 = feval(f, low);
y2 = feval(f, high);

% Edge case: if the function is negative at BOTH ends, the power constraint
% is not binding anywhere in [low, high] -- the unconstrained solution
% (dual_v -> 0, i.e. m = low) already satisfies the constraint. Return
% immediately instead of looping forever (original fallback below only
% handled y1<0 by shrinking low, with no corresponding fix when y2<0 too).
if y1 < 0 && y2 < 0
    disp('power(dual_v) < 0 at both ends -- constraint not binding, returning m = low.');
    m = low;
    return;
end

% Display error and finish if signs are not different
while (y1 * y2 > 0)
    disp('Have not found a change in sign. Will not continue...');
    disp(y1);disp(y2);
%m = 'Error';
if y1<0
        low = low*1e-1;
elseif y2>0
        high = high*1e1;
end
    y1 = feval(f, low);
    y2 = feval(f, high);

    % Re-check the negative-both-ends edge case after adjustment too,
    % in case the fallback still can't find a sign change.
    if y1 < 0 && y2 < 0 && low < 1e-12
        disp('Still cannot find sign change after shrinking low -- returning m = low.');
        m = low;
        return;
    end
%return;
end
% Work with the limits modifying them until you find
% a function close enough to zero.
%disp('Iter    low        high          x0');
while (abs(high - low) >= tol || feval(f, m) <= 0)
    i = i + 1;
% Find a new value to be tested as a root
    m = (high + low)/2;
    y3 = feval(f, m);
if y3 == 0
        fprintf('Root at x = %f \n\n', m);
return
end
%fprintf('%2i \t %f \t %f \t %f \n', i-1, low, high, m);
% Update the limits
if y1 * y3 > 0
        low = m;
        y1 = y3;
else
        high = m;
end
end
% Show the last approximation considering the tolerance
%w = feval(f, m);
% fprintf('\n x = %f produces f(x) = %f \n %i iterations\n', m, y3, i-1);
% fprintf(' Approximation with tolerance = %f \n', tol);