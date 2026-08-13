% Production of the heapmap of a matrix
function h = mat_heatmap(A,title_name)
blue = [linspace(0,1,128)' linspace(0,1,128)' ones(128,1)];
red = [ones(128,1) linspace(1,0,128)' linspace(1,0,128)'];
mycmap = [blue; red];

h = heatmap(A,Interpreter="latex");
title(title_name);
xlabel('Columns');
ylabel('Rows');
h.ColorbarVisible = 'on';
h.Colormap = mycmap;
clim([-max(abs(A(:))) max(abs(A(:)))]); % Set the color range
end
