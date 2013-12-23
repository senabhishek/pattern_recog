function [min_error, featsize_at_min_error, index] = find_minimum_error(error_matrix)
  min_error = 1;
  featsize_at_min_error = 2;
  index = 1;
  err_matrix_size = length(error_matrix);
  for i=1:err_matrix_size
    [temp_min_error, temp_min_featsize] = min(error_matrix(i).error);
    if (temp_min_error < min_error)
      min_error = temp_min_error;
      featsize_at_min_error = temp_min_featsize;
      index = i;
    end
  end
end

