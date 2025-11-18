import json
import datetime as dt
from docker.types import Mount
from airflow.models.dag import DAG
from airflow.models import Variable
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.utils.task_group import TaskGroup
from airflow.providers.docker.operators.docker import DockerOperator


with DAG(
        dag_id="RL_controller_DAG",
        description="Management of workflows.",
        start_date=dt.datetime.now(),
        schedule_interval=None,
        catchup=False,
        is_paused_upon_creation=True
) as dag:

    # ------------------------------------------------------------------------------------------------------------
    #                                      VARIABLES DEFINITION
    # ------------------------------------------------------------------------------------------------------------

    # directory path for mounting the docker volume
    host_path = Variable.get("host_path", deserialize_json=True)
    remote_path = "/opt/airflow/dags"

    # get configuration
    config = {
        "final_time": 14,  
        "time_start_checking_db": 5,
        "time_bw_check_db": 1,
        "runID": 623,
        "exp_ids": [exp_id for exp_id in range(19419, 19443)],
        "accelerated": True
    }

    # ------------------------------------------------------------------------------------------------------------
    #                                        BASE NODES DEFINITION
    # ------------------------------------------------------------------------------------------------------------

    def base_docker_node(task_id, command, retries=3, retry_delay=dt.timedelta(minutes=2), image="emulator2",
                        execution_timeout=dt.timedelta(minutes=10), trigger_rule='all_success'):
        
        return DockerOperator(
            task_id=task_id,
            image=image,
            auto_remove="force",
            working_dir=f"{remote_path}/scripts/controller_dag",
            command=command,
            mounts=[Mount(source=host_path, target=remote_path, type='bind')],
            mount_tmp_dir=False,
            network_mode="bridge",
            retries=0 if config["accelerated"] else retries,
            retry_delay=retry_delay,
            execution_timeout=execution_timeout,
            trigger_rule=trigger_rule 
        ) 
    

    # ------------------------------------------------------------------------------------------------------------
    #                                     RL CONTROLLER WORKFLOW
    # ------------------------------------------------------------------------------------------------------------

    # reset iteration
    start = base_docker_node(
        task_id="start",
        command=["python", "reset_iter.py"]
    )

    # create initial feed.json
    init_feed = base_docker_node(
        task_id="create_feeds",
        command=["python", "initial_feeds.py"]
    )

    # save actions in ilab db
    save_actions = base_docker_node(
        task_id=f"save_feeds",
        command=["python", "save_actions.py", str(config["runID"]), "feed.json", json.dumps(config["exp_ids"])]
    )

    # set dependencies
    start >> init_feed >> save_actions
    last_node = start

    # calculates iterations
    iterations = int((config["final_time"] - config["time_start_checking_db"]) / config["time_bw_check_db"])

    # iterates 
    for it in range(1, iterations + 1):

        time_wait = config['time_bw_check_db'] * (it - 1) + config['time_start_checking_db']

        # wait until next query (if accelerated, a gap of 0.25 minutes is added)
        wait = TimeDeltaSensor(
            task_id=f"{time_wait}_{'m' if config['accelerated'] else 'h'}_wait", 
            poke_interval=1 if config["accelerated"] else 30, 
            trigger_rule='all_done', 
            delta=dt.timedelta(minutes=(time_wait + 0.33)) if config["accelerated"] else dt.timedelta(hours=time_wait),
        )
        
        with TaskGroup(group_id=f"controller_{it}"):
        
            # query data from database:
            get_measurements = base_docker_node(
                task_id=f"get_measurements",
                command=["python", "query_and_save.py", str(config["runID"]), f"db_output.json"]
            )
                 
            # DOT controller
            DOT_controller = base_docker_node(
                task_id=f"RL_controller",
                image="sb3",
                command=["python", "RL_controller.py", f"db_output.json", "config.json", "feed.json"]
            )
            
            # save actions in ilab db
            save_actions = base_docker_node(
                task_id=f"save_feeds",
                command=["python", "save_actions.py", str(config["runID"]), "feed.json", json.dumps(config["exp_ids"])]
            )
        
            # set dependencies
            get_measurements >> DOT_controller >> save_actions


        # set dependencies
        last_node >> wait >> get_measurements           

        # save last node for next iteration
        last_node = wait
