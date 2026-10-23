"""Plot measured layout coordinates; kinematic and physics reports stay labelled."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    fig,axes=plt.subplots(1,len(args.reports),figsize=(5.2*len(args.reports),5.2),squeeze=False,layout='constrained')
    for ax,path in zip(axes[0],args.reports):
        report=json.loads(path.read_text());rows=report['layouts']
        for name,color in [('red_cube','#c83f46'),('blue_cube','#2561c4')]:
            points=np.array([r['layout']['objects'][name]['position'][:2] for r in rows
                             if name in r.get('layout',{}).get('objects',{})])
            if len(points):ax.scatter(points[:,0],points[:,1],color=color,label=f'{name} (n={len(points)})',s=34,alpha=.8,edgecolors='white',linewidths=.5)
        mode='Private geometry + IK admission' if report['mode']=='private_kinematics' else 'Genesis physics setup'
        ax.set_title(f"{report['robot']}\n{mode}",fontsize=11)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax.text(.02,.02,f"Seeds {report['seed_start']}–{report['seed_start']+len(rows)-1}",transform=ax.transAxes,fontsize=8)
    fig.suptitle('Randomized cube positions across the reachable desk workspace',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=180)


if __name__=='__main__':main()
