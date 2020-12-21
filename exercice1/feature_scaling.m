function scaled_data = feature_scaling(data)
  scaled_data = []
  for index= 1:size(data)(2)
    m = mean(data(:,index));
    std_dev = std(data(:,index));
    normalized = (data(:,index).- m)./std_dev;
    scaled_data = [scaled_data,normalized];
  endfor
endfunction
