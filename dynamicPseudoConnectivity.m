% Implementation of null model of Theiler et al. 1992, Physika D
% 17-01-2019 Jonathan Wirsich
% 18-01-2019 assure conjugate symmetry [Prichard 1994, Physical review letters] / [Theiler et al. 1992, Physika D]
% 
% @param ts - input timeseries(time x connections)
% @param iter - number of iterations
%
% @return shifted_ts1 - output timeseries:  iteration x connection x
% connectivity timecourse
function [shifted_ts1] = dynamicPseudoConnectivity(ts, iter)

    dim = size(ts);
    vec_size = dim(2);
    
    %preallocate results
    shifted_ts1 = zeros(iter, vec_size, dim(1)); 
    
    for i = 1:iter
    %get timeseries pair
       shift = rand(dim(1)/2, vec_size);
        
        count = 1;
        
        for r1 = 1:vec_size

                %FFT
                f1 = fft(ts(:,r1));
                %get phase
                phi1 = angle(f1);
                
                %randomize phase 
                phi_rand1(1)=0;
                phi_rand1(2:dim(1)/2+1) = wrapToPi(phi1(2:dim(1)/2+1)+(shift(:,count)*2*pi));
                %assure symmetry (Prichard 1994)
                phi_rand1(dim(1)/2+2:dim(1)) = -phi_rand1(dim(1)/2:-1:2);
                
                %ifft
                shifted_ts1(i,r1,:) = ifft(abs(f1).*exp(1i*phi_rand1'), 'symmetric');
        end
        
    end
    
end
