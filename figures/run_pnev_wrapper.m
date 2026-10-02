
try
    cd('/home/dylan/Documents/Github/fMCSI/figures');
    addpath(genpath(pwd));

    % Toolbox roots resolved on the Python side; empty when not found.
    cvx_root = '/home/dylan/Documents/MATLAB/cvx';
    caiman_root = '/home/dylan/Documents/Github/CaImAn-MATLAB';

    if ~isempty(cvx_root) && exist(cvx_root, 'dir') == 7
        addpath(genpath(cvx_root));
    end
    if ~isempty(caiman_root) && exist(caiman_root, 'dir') == 7
        addpath(genpath(caiman_root));
    end

    % Optional copy of cont_ca_sampler with the rise-time fix. Added last so it is
    % first on the path; empty unless the caller asked for it.
    fix_root = '';
    if ~isempty(fix_root)
        addpath(fix_root);
        fprintf('Using cont_ca_sampler with rise-time fix: %s\n', which('cont_ca_sampler'));
    end

    % Statistics Toolbox stand-ins, appended so a real installation wins.
    % Kept outside the genpath tree above so it cannot jump the queue.
    addpath('/tmp/omsi_matlab_shims', '-end');
    shimmed = {};
    for k = {'range', 'prctile', 'gamrnd', 'normrnd'}
        w = which(k{1});
        if ~isempty(strfind(w, '/tmp/omsi_matlab_shims'))
            shimmed{end+1} = k{1};
        end
    end
    if ~isempty(shimmed)
        fprintf('Statistics Toolbox not available for: %s -- using built-in stand-ins.\n', ...
                strjoin(shimmed, ', '));
    end

    % cvx_setup is slow and only needed once per session, so run it only when
    % cvx is present but not yet initialised.
    if exist('cvx_setup', 'file') == 2 && isempty(which('cvx_begin'))
        try
            cvx_setup;
        catch setup_err
            fprintf('cvx_setup failed: %s\n', setup_err.message);
        end
    end

    if isempty(which('cont_ca_sampler'))
        error('cont_ca_sampler not found on the MATLAB path. Set CAIMAN_MATLAB_HOME.');
    end

    load('/home/dylan/Documents/Github/fMCSI/figures/mcmc_input.mat');
    [n_cells, n_frames] = size(dff);

    params.Nsamples = n_sweeps;
    params.B = floor(n_sweeps / 2);
    if burn_in >= 0
        params.B = burn_in;
    end
    params.p = 2;
    params.f = fs;

    all_spikes = cell(n_cells, 1);
    all_ss = cell(n_cells, 1);
    all_init = cell(n_cells, 1);
    all_probs = zeros(n_cells, n_frames);
    model_traces = zeros(n_cells, n_frames);
    cell_times = nan(n_cells, 1);

    set(0, 'DefaultFigureVisible', 'off');
    fprintf('Running MCMC on %d cells...\n', n_cells);

    for i = 1:n_cells

        y = double(dff(i, :))';

        try
            if has_inits
                params.init = inits{i};
            elseif has_shifts
                % CaImAn's own start (the call cont_ca_sampler makes when no
                % init is given), with every spike moved by init_shifts(i) frames.
                p0 = params;
                p0.c = []; p0.b = []; p0.c1 = []; p0.g = []; p0.sn = []; p0.sp = [];
                p0.bas_nonneg = 0;
                SAM = get_initial_sample(y, p0);
                SAM.spiketimes_ = min(max(SAM.spiketimes_ + init_shifts(i), 1e-3), n_frames);
                params.init = SAM;
                all_init{i} = SAM.spiketimes_;
            end
            t_cell = tic;
            res = cont_ca_sampler(y, params);
            cell_times(i) = toc(t_cell);

            samples = res.ss;
            n_post = length(samples);

            prob_trace = zeros(1, n_frames);
            for s = 1:n_post
                st = samples{s};
                if ~isempty(st)
                    idx = round(st);
                    idx = idx(idx >= 1 & idx <= n_frames);
                    prob_trace(idx) = prob_trace(idx) + 1;
                end
            end
            all_probs(i, :) = prob_trace / max(1, n_post);

            if ~isempty(samples)
                all_spikes{i} = samples{end};
            end
            if save_samples
                all_ss{i} = samples;
            end

            temp_trace = make_mean_sample(res, y);
            model_traces(i, :) = temp_trace(:)';

        catch ME
            fprintf('Error on cell %d: %s\n', i, ME.message);
        end
    end

    if save_samples
        save('/home/dylan/Documents/Github/fMCSI/figures/mcmc_output.mat', 'all_spikes', 'all_probs', 'model_traces', 'all_ss', 'all_init');
    else
        save('/home/dylan/Documents/Github/fMCSI/figures/mcmc_output.mat', 'all_spikes', 'all_probs', 'model_traces');
    end
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\n', ME.message);
    exit(1);
end
