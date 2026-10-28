"""Concrete prepared MadNLP workspace. Structural options are fixed at setup.
`solve!` restarts from u0, clears per-fit counters/history, and returns borrowed
solution buffers. Copy results that must survive another solve. No warm-start promise.
"""
struct PreparedMadNLPCache{C,S,R} <: SciMLBase.AbstractOptimizationCache
    cache::C
    solver::S
    stats::R
end
function init_prepared(prob, opt::MadNLPOptimizer; kwargs...)
    cache = SciMLBase.init(prob,opt;kwargs...)
    args=cache.solver_args
    maxiters=OptimizationBase._check_and_convert_maxiters(args.maxiters)
    maxtime=OptimizationBase._check_and_convert_maxtime(args.maxtime)
    meta,options=__map_optimizer_args(cache,opt;abstol=args.abstol,reltol=args.reltol,
        maxiters,maxtime,verbose=get(args,:verbose,false),progress=cache.progress,callback=cache.callback)
    solver=MadNLP.MadNLPSolver(NLPModelsAdaptor(cache,meta,NLPModels.Counters());options...)
    solver.kkt.quasi_newton isa MadNLP.CompactLBFGS || throw(ArgumentError("prepared reset supports CompactLBFGS"))
    solver.opt.barrier isa MadNLP.MonotoneUpdate || throw(ArgumentError("prepared reset supports MonotoneUpdate"))
    PreparedMadNLPCache(cache,solver,MadNLP.MadNLPExecutionStats(solver))
end
function SciMLBase.reinit!(pc::PreparedMadNLPCache;u0=pc.cache.u0,p=pc.cache.p)
    length(u0)==length(pc.solver.nlp.meta.x0) || throw(DimensionMismatch("structural changes require init_prepared"))
    SciMLBase.reinit!(pc.cache;u0,p)
    copyto!(pc.solver.nlp.meta.x0,u0)
    cb=pc.solver.intermediate_callback
    pc.solver.intermediate_callback=MadNLPProgressLogger(cb.callback,cb.progress,cb.maxiters,p)
    pc
end

function SciMLBase.solve!(p::PreparedMadNLPCache)
    MadNLP.reset!(p.solver)
    r=MadNLP.solve!(p.solver,p.stats)
    stats=OptimizationBase.OptimizationStats(time=r.counters.total_time,iterations=r.iter,
        fevals=r.counters.obj_cnt,gevals=r.counters.obj_grad_cnt)
    SciMLBase.build_solution(p.cache,p.cache.opt,r.solution,r.objective;original=r,
        retcode=map_madnlp_status(r.status),stats)
end
