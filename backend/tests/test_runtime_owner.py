"""Independent processes cannot silently create a second Genesis owner."""
import subprocess
import sys


def test_process_lease_blocks_a_second_owner_and_releases_on_exit(tmp_path):
    path = str(tmp_path/'world.lock')
    source = 'from backend.app.simulation.runtime_owner import acquire_genesis_owner; import sys; acquire_genesis_owner(sys.argv[1]);'
    owner = subprocess.Popen([sys.executable,'-u','-c',source+'print("ready",flush=True); input()',path],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert owner.stdout.readline().strip()=='ready'
        competing = subprocess.run([sys.executable,'-c',source,path],capture_output=True,text=True,timeout=10)
        assert competing.returncode!=0 and 'owns the world' in competing.stderr
    finally:
        owner.communicate('\n',timeout=10)
    restarted = subprocess.run([sys.executable,'-c',source,path],capture_output=True,text=True,timeout=10)
    assert restarted.returncode==0,restarted.stderr
