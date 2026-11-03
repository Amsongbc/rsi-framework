"""Known-answer analysis and counterbalancing checks before data collection."""
import json
from pathlib import Path
import numpy as np
from analyze_efficiency import stats,exact_p
from study_plan import ROOT,build


def main():
    cfg=json.loads((ROOT/'experiment_config.json').read_text());decl=build(cfg)
    available=[]
    for layout in cfg['layouts']:
        assert layout not in {0,1,2,3,4,5}
        for seed in cfg['eval_seeds']:
            p=ROOT.parent/'RoboDojo-resources/Assets/Eval_Layout/RoboDojo/arx_x5'/str(seed)/f'classify_objects_{layout}.json'
            assert p.is_file();available.append(str(p))
    assert exact_p(6,0)==.03125 and exact_p(2,0)==.5 and exact_p(0,0)==1.
    blocks=[]
    for b in decl['blocks']:
        row={k:b[k] for k in ['seed','layout','case_id']}
        for arm,f in [('original',1.),('sparse',.8),('compact',.5),('combined',.4)]:
            row[arm]={'success':True,'score':1.,'control_min':10*f,'tokens':1000*f,
                'uncached_tokens':100*f,'api_calls':10*f,'native_actions':100*f}
        blocks.append(row)
    result=stats(blocks,cfg['layouts'])
    assert np.isclose(result['comparisons_vs_original']['combined']['tokens']['difference'],-60)
    assert np.allclose(result['comparisons_vs_original']['combined']['tokens']['cluster_95'],[-60,-60])
    assert np.isclose(result['factorial_log_resource_effects']['tokens']['sparse']['geometric_percent_effect'],-20)
    assert np.isclose(result['factorial_log_resource_effects']['tokens']['compaction']['geometric_percent_effect'],-50)
    assert np.isclose(result['factorial_log_resource_effects']['tokens']['interaction']['geometric_percent_effect'],0)
    assert result['combined_passes_resource_and_observed_success_screen']
    report={'passed':True,'benchmark_episode':False,'layout_files_verified':available,
        'checks':['48unique predeclaredcases','12perarm','6episodes perarm perlane andwave',
            '2episodes perarm perseed perlane andwave','all6concurrentarm pairings occur4times',
            'holdout disjoint from layouts0-5','known-answer resource ratios andfactor effects','exact paired pknown answers']}
    (ROOT/'offline_plan_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='layout_files_verified'}))


if __name__=='__main__':main()
