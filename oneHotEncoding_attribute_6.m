% this function was originally written by reference 1 on our poster. we modify
% the code for conversion of 6 high level variables to numberic values.
% (i.e. not feature engineering to 21 new variable but only applying
% numerical values to the 6 initial variables and subattributes.

function [ encoded ] = oneHotEncoding_attribute( raw )
%oneHotEncoding_attribute
%   converts categorical features to binary features
%Input:
%   raw - a 1x6 cell of data point containing 6 categorical features.
%Output:
%   encoded - a 1x21 binary vector of converted features.

% From the description of dataset,
% total number of binary features from 6 attributes is 4+4+4+3+3+3 = 21.
encoded = zeros(1, 6);

% Attribute - buying: vhigh, high, med, low
% converts to index 1, 2, 3, 4 respectively.

switch raw{1}
    case 'vhigh'
        encoded(1) = 4;
    case 'high'
        encoded(1) = 3;
    case 'med'
        encoded(1) = 2;
    case 'low'
        encoded(1) = 1;
    otherwise
        error(strcat(raw{1}, ' is not in buying attributes'));
end

% Atrribute - maint: vhigh, high, med, low
% converts to index 5, 6, 7, 8 respectively.
switch raw{2}
    case 'vhigh'
        encoded(2) = 4;
    case 'high'
        encoded(2) = 3;
    case 'med'
        encoded(2) = 2;
    case 'low'
        encoded(2) = 1;
    otherwise
        error(strcat(raw{2}, ' is not in maint attributes'));
end

% Attribute - doors: 2, 3, 4, 5more
% converts to index 9, 10, 11, 12 respectively.
switch raw{3}
    case '2'
        encoded(3) = 1;
    case '3'
        encoded(3) = 2;
    case '4'
        encoded(3) = 3;
    case '5more'
        encoded(3) = 4;
    otherwise
        error(strcat(raw{3}, ' is not in doors attributes'));
end

% Attribute - persons: 2, 4, more
% converts to index 13, 14, 15 respectively.
switch raw{4}
    case '2'
        encoded(4) = 1;
    case '4'
        encoded(4) = 2;
    case 'more'
        encoded(4) = 3;
    otherwise
        error(strcat(raw{4}, ' is not in persons attributes'));
end

% Attribute - lug_boot: small, med, big
% converts to index 16, 17, 18 respectively.
switch raw{5}
    case 'small'
        encoded(5) = 1;
    case 'med'
        encoded(5) = 2;
    case 'big'
        encoded(5) = 3;
    otherwise
        error(strcat(raw{5}, ' is not in lug_boot attributes'));
end

% Attribute - safety: low, med, high
% converts to index 19, 20, 21 respectively.
switch raw{6}
    case 'low'
        encoded(6) = 1;
    case 'med'
        encoded(6) = 2;
    case 'high'
        encoded(6) = 3;
    otherwise
        error(strcat(raw{6}, ' is not in safety attributes'));
end
end