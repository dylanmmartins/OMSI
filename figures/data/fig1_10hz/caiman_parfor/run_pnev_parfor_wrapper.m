
try
    cd('C:/Users/dmartins/Documents/GitHub/OMSI/figures/data/fig1_10hz/caiman_parfor');

    cvx_root = 'C:/Users/dmartins/Desktop/cvx';
    caiman_root = 'C:/Users/dmartins/Documents/Github/CaImAn-MATLAB';
    if ~isempty(cvx_root) && exist(cvx_root, 'dir') == 7
        addpath(genpath(cvx_root));
    end
    if ~isempty(caiman_root) && exist(caiman_root, 'dir') == 7
        addpath(genpath(caiman_root));
    end

    % Statistics Toolbox stand-ins, appended so a real installation wins.
    addpath('C:/Users/dmartins/AppData/Local/Temp/omsi_matlab_shims', '-end');

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
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('Parallel Computing Toolbox is not licensed; parfor would run serially.');
    end

    load('C:/Users/dmartins/Documents/GitHub/OMSI/figures/data/fig1_10hz/caiman_parfor/mcmc_input.mat');
    [n_cells, n_frames] = size(dff);

    params.Nsamples = n_sweeps;
    params.B = floor(n_sweeps / 2);
    params.p = 2;
    params.f = fs;

    all_spikes = cell(n_cells, 1);
    all_probs = zeros(n_cells, n_frames);
    model_traces = zeros(n_cells, n_frames);
    cell_times = nan(n_cells, 1);

    set(0, 'DefaultFigureVisible', 'off');

    % Fresh pool so start-up is timed and the worker count is the one asked for.
    % The client path set above is passed on to the workers.
    delete(gcp('nocreate'));
    t_pool = tic;
    if n_workers > 0
        pool = parpool('Processes', n_workers);
    else
        pool = parpool('Processes');
    end
    pool_time = toc(t_pool);
    worker_threads = fetchOutputs(parfevalOnAll(pool, @maxNumCompThreads, 1));
    fprintf('POOL workers=%d start=%.2fs threads/worker=%s\n', ...
            pool.NumWorkers, pool_time, mat2str(worker_threads(:)'));

    % cont_ca_sampler initializes with constrained_foopsi, whose default
    % method is CVX, as in the serial run. Make sure every worker sees the
    % same CVX and solver as the client instead of failing cell by cell.
    worker_cvx = fetchOutputs(parfevalOnAll(pool, @() which('cvx_begin'), 1), 'UniformOutput', false);
    if any(cellfun(@isempty, worker_cvx))
        error('cvx_begin not on the path of every worker.');
    end
    fprintf('CVX client: %s  solver=%s\n', which('cvx_begin'), cvx_solver);
    worker_solver = fetchOutputs(parfevalOnAll(pool, @cvx_solver, 1), 'UniformOutput', false);
    fprintf('CVX workers: %s  solvers=%s\n', worker_cvx{1}, strjoin(unique(worker_solver), ','));

    fprintf('Running MCMC on %d cells with parfor...\n', n_cells);
    t_loop = tic;
    parfor i = 1:n_cells

        y = double(dff(i, :))';

        try
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

            temp_trace = make_mean_sample(res, y);
            model_traces(i, :) = temp_trace(:)';

        catch ME
            fprintf('Error on cell %d: %s\n', i, ME.message);
        end
    end
    loop_time = toc(t_loop);
    n_workers_used = pool.NumWorkers;
    fprintf('TIMING parfor=%.2fs sum_per_cell=%.2fs failed_cells=%d\n', ...
            loop_time, sum(cell_times, 'omitnan'), sum(isnan(cell_times)));

    delete(pool);
    save('C:/Users/dmartins/Documents/GitHub/OMSI/figures/data/fig1_10hz/caiman_parfor/mcmc_output.mat', 'all_spikes', 'all_probs', 'model_traces', 'cell_times', ...
         'pool_time', 'loop_time', 'n_workers_used', 'worker_threads');
    exit(0);

catch ME
    fprintf('Global MATLAB Error: %s\n', ME.message);
    exit(1);
end
