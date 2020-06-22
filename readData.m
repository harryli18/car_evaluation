function [ data, label ] = readData( d )
%This code converts the one initial cell containing catrgorical predictor variables into a comma
%seperated format, it then applies one hot encoding to the class labels for
%modelling purposes.

len = size(d, 1);

data = cell(len, 6);
label = zeros(len, 1);
K = 4;

% splits and encodes data for each row
for i = 1:len
    temp = strsplit(d{i}, ',');
    data(i, :) = temp(1:6);
    label(i) = oneHotEncoding_label(temp{7});
end

end

