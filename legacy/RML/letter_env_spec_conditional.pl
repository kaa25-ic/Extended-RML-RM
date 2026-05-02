:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).
match(_event, a_match) :- deep_subdict(_{'a':T}, _event), T=1.0.
match(_event, b_match) :- deep_subdict(_{'b':T}, _event), T=1.0.
match(_event, c_match) :- deep_subdict(_{'c':T}, _event), T=1.0.
match(_event, d_match) :- deep_subdict(_{'d':T}, _event), T=1.0.
match(_event, not_abcd) :- not(match(_event, a_match)), not(match(_event, b_match)), not(match(_event, c_match)), not(match(_event, d_match)).
match(_, any).
trace_expression('Main', Main) :- Main=(star((not_abcd:eps))*app(A, [0])), A=gen(['n'], ((a_match:eps)*(star((not_abcd:eps))*(app(A, [(var('n')+1)])\/app(B, [(var('n')+1)]))))), B=gen(['n'], ((b_match:eps)*app(C, [var('n')]))), C=gen(['n'], guarded((var('n')<2.5), (star((not_abcd:eps))*(c_match:eps)), (star((not_abcd:eps))*(d_match:eps)))).
