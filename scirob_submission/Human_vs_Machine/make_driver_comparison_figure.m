% Neural Network Vehicle Models for High-Performance Automated Driving
% Nathan A. Spielberg, Matthew Brown, Nitin R. Kapania, John C. Kegelman, J. Christian Gerdes

%% Dependencies
% This script requires the following packages:
%
% ds2nfu.m
% Author: Michelle Hirsch
% Available from MATLAB Central File Exchange:
% https://www.mathworks.com/matlabcentral/fileexchange/10656-data-space-to-figure-units-conversion
%
% export_fig
% Authors: Oliver J. Woodford, Yair M. Altman
% Available from GitHub:
% https://github.com/altmany/export_fig

%% Constants and default figure properties

% figure dimensions
FIG_X = 5; FIG_Y = 1; % [in]
TEXT_WIDTH = 5.9; % [in]

% section indices
TURN1    = 2; % start of Turn 1 --> sectionIdx(2)
TURN2    = 3;
TURNS3_4 = 4;
TURN5    = 5;
TURN6    = 6; % start of Turn 6 --> sectionIdx(6)

% colors
LIGHT_GRAY = [192 192 192]/255;
DARK_GRAY  = [ 84  84  84]/255;
REDS = [0.9412, 0.8471, 0.8471; ...
        0.8353, 0.5804, 0.5804; ...
        0.7333, 0.3608, 0.3608; ...
        0.6235, 0.2510, 0.2510; ...
        0.5137, 0.1373, 0.1412; ...
        0.4392, 0.0863, 0.0902; ...
        0.3647, 0.0314, 0.0353; ...
        0.3059, 0.0118, 0.0157];
BLUES = [0.9686, 0.9843, 1.0000; ...
         0.8706, 0.9216, 0.9686; ...
         0.7765, 0.8588, 0.9373; ...
         0.6196, 0.7922, 0.8824; ...
         0.4196, 0.6824, 0.8392; ...
         0.2588, 0.5725, 0.7765; ...
         0.1294, 0.4431, 0.7098; ...
         0.0314, 0.2706, 0.5804];

% figure properties
set(groot, 'DefaultFigureUnits'      , 'inches', ...
           'DefaultFigureColor'      , 'white' , ...
           'DefaultFigureNumberTitle', 'off'   , ...
           'DefaultAxesNextPlot'     , 'add'   , ...
           'DefaultAxesFontSize'     , 7       , ...
           'DefaultAxesFontName'     , 'Arial' , ...
           'DefaultAxesLabelFontSize', 1.0     , ...
           'DefaultTextFontSize'     , 7       , ...
           'DefaultTextFontName'     , 'Arial');

AXES_PROPS = {'Box'    , 'off' , ...
              'TickDir', 'out'};

BOX_PROPS = {'boxstyle'   , 'outline', ...
             'notch'      , 'on'     , ...
             'outliersize', 3        , ...
             'symbol'     , 'o'     };

LINE_PROPS = {'Color'    , 'black', ...
              'LineWidth', 1.5   };

ARROW_PROPS = [LINE_PROPS  ,      ...
              {'HeadLength', 6  , ...
               'HeadWidth' , 6}];

TEXT_PROPS = {'EdgeColor'          , 'none'   , ...
              'HorizontalAlignment', 'center' , ...
              'VerticalAlignment'  , 'middle'};

EXPORT_FIG_OPTS = {'-nocrop', '-painters', '-depsc'};

%% Load data
% loads 'track', 'section', and 'driver' structs
load('driver_comparison_data.mat');

% Data description
% driver: 1×2 struct array with fields:
%                id: Driver ID
%      section_time: 5x10 array of the driver's section times for each
%                    of the five track sections and each of the ten
%                    trials.
% mean_section_time: 5x1 array of the driver's mean section times.
%   mad_median_path: Mean absolute deviation from the driver's median
%                    path as measured by median lateral deviation from
%                    the track centerline.
%  mad_desired_path: Mean absolute deviation from Shelley's nominal
%                    racing line.
%             trial: 1x10 struct array corresponding to each driver's
%                    trials with fields:
%                  time: Time in seconds since the beginning of the trial.
%               path_EN: East-North coordinates of the driven path in
%                        meters from the GPS reference point.
%              distance: Distance traveled in meters since the beginning of
%                        the trial.
%                 speed: Vehicle speed in meters per second.
%
% section: struct with fields:
%      centerlineEN: East-North coordinates of track centerline.
%    centerlineDist: Distance along the track centerline.
%        sectionIdx: Indices corresponding to the start of each test
%                    section.
%          insideEN: East-North coordinates of the inside track boundary.
%         outsideEN: East-North coordinates of the outside track boundary.
%              Time: 5x20 array of the drivers' section times for each of
%                    the five track sections, each of the ten trials, and
%                    each of the two driver.
%            Driver: The driver ID corresponding to each column of the
%                    section Time array.
%
% track: struct with fields:
%              name: Name of the test track.
%           xyz_ref: Earth-Centered-Earth-Fixed (ECEF) reference point used
%                    to convert latitude, longitude, altitude coordinates
%                    to East, North, Up coordinates.
%          insideEN: East-North coordinates of the inside track boundary.
%         outsideEN: East-North coordinates of the outside track boundary.
%      centerlineEN: East-North coordinates of the track centerline.
%          sections: struct with the name and GPS reference point for each
%                    track section.
%              dist: Distance along the track centerline.
%        sectionIdx: Indices corresponding to the start of each test
%                    section.

%% Figures 2B, 2C, and 2D.
% B. The human driver’s mean absolute deviation from the median (MAD
% median) path projected onto the first five turns of Thunderhill Raceway
% in Willows, CA. C. Shelley’s MAD median path scaled by a factor of four
% to highlight relative differences. D. MAD median path for both the human
% driver and Shelley (in red) along with Shelley’s mean absolute deviation
% from the intended path (in blue).

figure_pos_graph = [FIG_X FIG_Y     TEXT_WIDTH .3*TEXT_WIDTH];
figure_pos_map   = [FIG_X FIG_Y .24*TEXT_WIDTH .3*TEXT_WIDTH];
axes_pos_graph = [.09 .23 .89 .75];
axes_pos_map   = [0 0 1 1];
num_points = 2000;
mag = 25;

text_graph = {sprintf('Human\nlateral dispersion')  ; ...
              sprintf('Shelley\nlateral dispersion'); ...
              sprintf('Shelley\nlateral error')};

label_pos_x = [930, 895;
                 0,   0;
               573, 585;
               725, 690];
label_pos_y = [0.55, 0.55;
               0   , 0   ;
               0.22, 0.03;
               0.27, 0.27];

% axes limits
x_lim_graph = [400 1450];
x_lim_map   = [5  410];
y_lim_graph = [0 .875];
y_lim_map = [-325 450];
c_lim = [.25 .45;
         .05 .10];

c_divs    = length(REDS);
red_cmap  = flipud(REDS);
blue_cmap = repmat(BLUES(6,:), c_divs,1);

north_length = 100;
north_offset = [-.025 .005 .05 .05];
north_x      =  325;
north_y      =  300;

scale_length = 100;
scale_offset = [-.050 .005 .10 .05];
scale_x      = 275;
scale_y      = 175;

turn_pos = [ 20 -180; ...
            370 -210; ...
            155    0; ...
            190  140; ...
            116  344];
halign = {'left', 'right', 'left', 'right', 'left'};

% calculate section dividers
div = NaN(2,2,3);
for idxS = TURN2 : TURN5
    ii = track.sectionIdx(idxS);
    v = diff(track.centerlineEN(ii:ii+1,:));
    p = [-v(2) v(1)];
    p = p/norm(p);
    div(1,:,idxS-TURN2+1) = track.centerlineEN(ii,:) + mag*p;
    div(2,:,idxS-TURN2+1) = track.centerlineEN(ii,:) - mag*p;
end

hf_graph = figure('Name', 'Human Shelley Path Dispersion', ...
                  'Position', figure_pos_graph);
ha_graph = axes('Parent', hf_graph);
xlabel(ha_graph, 'Distance along centerline (m)');
hylbl = ylabel(ha_graph, 'Mean absolute deviation (m)');
set(ha_graph, 'Position'  , axes_pos_graph, ...
              'XLim'      , x_lim_graph   , ...
              'YLim'      , y_lim_graph   , ...
              'XMinorTick', 'on'          , ...
              'YMinorTick', 'on'          , ...
              AXES_PROPS{:});

y_ticks = get(ha_graph, 'YTick');
y_tick_labels = cell(size(y_ticks));
for ii = 1 : length(y_ticks)
    y_tick_labels{ii} = sprintf('%4.2f',y_ticks(ii));
end
set(ha_graph, 'YTickLabel', y_tick_labels);
drawnow;

hylbl_pos = hylbl.Position;
hylbl.Position = hylbl_pos + [0 -0.05*diff(y_lim_graph) 0];

pbar = get(ha_graph, 'PlotBoxAspectRatio');
dar  = get(ha_graph, 'DataAspectRatio');
ar_graph   = (pbar(2)/pbar(1)) / (dar(2)/dar(1));

graph_edges = linspace(c_lim(1,1), c_lim(1,2), c_divs+1);
graph_edges(1) = -inf; graph_edges(end) = inf;
for dd = 1 : length(driver)
    % Add MAD median path vs. distance along centerline to graph.
    X = section.centerlineDist;
    Y = driver(dd).mad_median_path;
    C = driver(dd).mad_median_path;

    % evenly spaced points
    x = 1:length(X);
    y = [0; cumsum(sqrt(diff(X).^2+diff(Y*ar_graph).^2))];
    yi = (0:num_points-1)*y(end)/(num_points-1);
    yi(end) = y(end);
    xi = round(interp1(y,x,yi));

    X = X(xi);
    Y = Y(xi);
    C = C(xi);

    [~,~,bin] = histcounts(C, graph_edges);
    for ii = c_divs : -1 : 1
        idx = (bin == ii);
        plot(ha_graph, X(idx), Y(idx), ...
            'Color'     , red_cmap(ii,:), ...
            'LineStyle' , 'none'    , ...
            'Marker'    , '.'       , ...
            'MarkerSize', 4        );
    end

    % Plot MAD median path projected along trackmap.
    hf_map = figure('Name', [driver(dd).id ' Path Dispersion Map'], ...
                    'Position', figure_pos_map);
    ha_map = axes('Parent', hf_map);
    colormap(ha_map, red_cmap);
    axis(ha_map, 'equal', 'off');
    set(ha_map, 'Box'     , 'off'       , ...
                'Position', axes_pos_map, ...
                'XTick'   , []          , ...
                'XLim'    , x_lim_map   , ...
                'YTick'   , []          , ...
                'YLim'    , y_lim_map  );
    drawnow;

    pbar = get(ha_map, 'PlotBoxAspectRatio');
    dar  = get(ha_map, 'DataAspectRatio');
    ar_map   = (pbar(2)/pbar(1)) / (dar(2)/dar(1));

    X = section.centerlineEN(:,1);
    Y = section.centerlineEN(:,2);
    C = driver(dd).mad_median_path;

    % evenly spaced points
    x = 1 : length(X);
    y = [0; cumsum(sqrt(diff(X).^2+diff(Y*ar_map).^2))];
    yi = (0:num_points-1)*y(end)/(num_points-1);
    yi(end) = y(end);
    xi = round(interp1(y,x,yi));

    X = X(xi);
    Y = Y(xi);
    C = C(xi);

    map_edges = linspace(c_lim(dd,1), c_lim(dd,2), c_divs+1);
    map_edges(1) = -inf; map_edges(end) = inf;
    [~,~,bin] = histcounts(C, map_edges);
    for ii = c_divs : -1 : 1
        idx = (bin == ii);
        plot(ha_map, X(idx), Y(idx), ...
            'Color'     , red_cmap(ii,:), ...
            'LineStyle' , 'none'        , ...
            'Marker'    , '.'           , ...
            'MarkerSize', 8            );
    end

    % add section dividers
    for idxS = 1 : size(div,3)
        plot(div(:,1,idxS), div(:,2,idxS), ...
            'Color'    , 'black', ...
            'LineStyle', '-'    , ...
            'LineWidth', .8    );
    end

    % label turns
    for idxT = 1 : size(turn_pos,1)
        text(ha_map, turn_pos(idxT,1), turn_pos(idxT,2), ...
            sprintf('Turn %d', idxT), ...
            'HorizontalAlignment', halign{idxT});
    end

    % annotate North and scale
    [north_x_norm(1), north_y_norm(1)] = ds2nfu(ha_map, north_x, north_y);
    [north_x_norm(2), north_y_norm(2)] = ds2nfu(ha_map, north_x, north_y+north_length);
    [scale_x_norm(1), scale_y_norm(1)] = ds2nfu(ha_map, scale_x, scale_y);
    [scale_x_norm(2), scale_y_norm(2)] = ds2nfu(ha_map, scale_x+scale_length, scale_y);

    annotation('arrow', north_x_norm, north_y_norm, ARROW_PROPS{:});
    annotation('textbox', 'Position', [north_x_norm(2) north_y_norm(2) 0 0] + north_offset, ...
                          'String', 'N', ...
                          TEXT_PROPS{:});
    annotation('textbox', 'Position', [mean(scale_x_norm) scale_y_norm(2) 0 0] + scale_offset, ...
                          'String', '100m', ...
                          TEXT_PROPS{:});
    annotation('line', scale_x_norm, scale_y_norm, LINE_PROPS{:});

    text(ha_map, x_lim_map(1) + 0.03*diff(x_lim_map), ...
                 y_lim_map(1) + 0.75*diff(y_lim_map), ...
                 driver(dd).id, ...
                 TEXT_PROPS{:});

    drawnow;
    export_fig(hf_map, [driver(dd).id 'DispersionMap'], '-pdf', ...
            EXPORT_FIG_OPTS{:});
    export_fig(hf_map, [driver(dd).id 'DispersionMap'], '-eps', ...
            EXPORT_FIG_OPTS{:});
end

% Add Shelley's lateral error vs. distance along centerline to graph.
X = section.centerlineDist;
Y = driver(2).mad_desired_path;
C = driver(2).mad_desired_path;

% evenly spaced points
num_points = num_points*4;
x = 1:length(X);
y = [0; cumsum(sqrt(diff(X).^2+diff(Y*ar_graph).^2))];
yi = (0:num_points-1)*y(end)/(num_points-1);
yi(end) = y(end);
xi = round(interp1(y,x,yi));

X = X(xi);
Y = Y(xi);
C = C(xi);

[~,~,bin] = histcounts(C, graph_edges);
for ii = c_divs : -1 : 1
    idx = (bin == ii);
    plot(ha_graph, X(idx), Y(idx), ...
        'Color'     , blue_cmap(ii,:), ...
        'LineStyle' , 'none'    , ...
        'Marker'    , '.'       , ...
        'MarkerSize', 4        );
end

for idxS = TURN2 : TURN5
    plot(ha_graph, track.dist(track.sectionIdx(idxS))*[1 1], [-1e6 1e6], ...
        'Color', 'black', ...
        'LineStyle', '--', ...
        'LineWidth', 0.8);
    text(ha_graph, track.dist(track.sectionIdx(idxS))+20, ...
                   y_lim_graph(2)-diff(y_lim_graph)*.03, ...
                   track.sections(idxS).name);
end

drawnow;
export_fig(hf_graph, 'HumanShelleyDispersion', '-pdf', EXPORT_FIG_OPTS{:});
export_fig(hf_graph, 'HumanShelleyDispersion', '-eps', EXPORT_FIG_OPTS{:});

%% Figure 2E.
% Segment times from champion amateur driver benchmarked to Shelley.

figure_pos = [FIG_X FIG_Y TEXT_WIDTH .3*TEXT_WIDTH];
axes_pos   = [.100 .14 .255 .83 ; ...
              .415 .14 .255 .83 ; ...
              .730 .14 .255 .83];

y_lim      = [13.1 13.7; ...
              13.8 14.4; ...
              12.1 12.7];
y_step     = 0.1;

sections   = [TURN2 TURNS3_4 TURN5];
title_text = {'Turn 2', 'Turns 3 & 4', 'Turn 5'};
title_pos  = [13.65; 14.35; 12.65];

hf = figure('Name', 'Segment Time boxplot', ...
            'Position', figure_pos);
ha = nan(length(sections),1);
for idxS = 1 : length(sections)
    ha(idxS) = axes('Parent', hf);
    if (idxS==1)
        ylabel(ha(idxS), 'Segment time (s)');
    end
    boxplot(ha(idxS), section.Time(sections(idxS),:), section.Driver, ...
        BOX_PROPS{:});

    y_ticks = y_lim(idxS,1) : y_step : y_lim(idxS,2);
    y_tick_labels = cell(size(y_ticks));
    for idx = 1 : length(y_ticks)
        y_tick_labels{idx} = sprintf('%4.1f', y_ticks(idx));
    end

    set(ha(idxS), 'Position'  , axes_pos(idxS,:), ...
                  'YLim'      , y_lim(idxS,:)   , ...
                  'YTick'     , y_ticks         , ...
                  'YTickLabel', y_tick_labels   , ...
                  AXES_PROPS{:});

    ht = title(title_text{idxS});
    current_title_pos = get(ht, 'Position');
    current_title_pos(2) = title_pos(idxS);
    set(ht, 'Position'  , current_title_pos, ...
            'FontWeight', 'normal');

    % adjust boxplot elements
    tags = {'Box', 'Median', 'Upper Whisker', 'Lower Whisker', ...
            'Upper Adjacent Value', 'Lower Adjacent Value'};
    for idxT = 1 : length(tags)
        set(findobj(ha(idxS), 'tag', tags{idxT}), ...
            'Color'    , 'black', ...
            'LineStyle', '-'   );
    end

    ho = findobj(findobj(ha(idxS), 'tag', 'Outliers'));
    for idx = 1 : length(ho)
        idxD = get(ho(idx), 'XData');
        if ~isnan(idxD)
            set(ho(idx), 'MarkerEdgeColor', DARK_GRAY);
        end
    end

    % fill box interior
    hb = findobj(ha(idxS), 'tag', 'Box');
    for idx = 1 : length(hb)
        xb = get(hb(idx), 'XData');
        yb = get(hb(idx), 'YData');
        hp = patch(xb, yb, LIGHT_GRAY);
        uistack(hp, 'bottom');
    end

    % mark mean
    for idxD = 1 : length(driver)
        plot(ha(idxS), idxD, driver(idxD).mean_section_time(sections(idxS)), ...
            'Marker'         , 'd'     , ...
            'MarkerSize'     , 2       , ...
            'MarkerEdgeColor', 'black' , ...
            'MarkerFaceColor', 'black');
    end
end

drawnow;
export_fig('secTimeBoxPlot', '-pdf', EXPORT_FIG_OPTS{:});
export_fig('secTimeBoxPlot', '-eps', EXPORT_FIG_OPTS{:});