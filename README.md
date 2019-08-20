# qmmm - library

## Installation procedure
Please follow to steps in order to make the`qmmm` library working using your own user account.
Since the library will log into the cluster for you have to provide it with you private key RSA file and your username.
This part of the configuration cannot be preconfigured

 1. **Obtain configuration files**
    1. Clone the [`qmmm-configuration`](http://193.171.82.188:8189/dnoeger/qmmm-configuration) repository from our [GitLab](http://193.171.82.188:8189) server.
       <br/> `git clone http://imw-srv-188.unileoben.ac.at:8189/cms/cms-notebooks.git`
       <br /> Please *remember* the directory where you cloned the repository to!
    2. Check out the the right branch in your local repository.
       The branch name is `configuration-cluster-HOSTNAME` e. g. `configuration-cluster-smmpmech` 
       for [smmpmech](smmpmech.uileoben.ac.at) cluster.
       <br /> Therefore enter the following command to your terminal
       <br /> `cd qmmm-configuration`
       <br /> `git checkout <branchname>`
 2. **Adapt configuration files for your needs**
    1. The file `application.json` defines the execution preambles as well the path to binaries
    2. Adapt `remote.json` in `qmmm-configuration` as shown in the [example configuration](#remotes) section
### Example configurations

#### remotes
The file `remote.json` specifies how the connection to remote has to be initiated and the queing system if there 
is any. Furthermore it defines the the partitions as well as default values for the queueing system.

For each remote the following keys must be specified:
  - `user`: username
  - `identity_file`: private key file
  - `host`: host name
  - `prefix`: calculation directory prefix
  - `partitions`: partitions (a list of partition objects)
    - `id`: id (a list partition aliases)
    - `name`: name of the partition defined by the queueing system
    - `qos`: Quality of service string
    - `default`: default values passed on to the queuing system
        - `tasks_per_node`: number of processes per node
        - `exclusive`: wether to allow other jobs on the reserved nodes
        - `nodes`: number of nodes
        - `interactive`: wether to run the jobs interactively
        - `mem`: maximum memory limit
        
```json5
{
  "remote1": {
    "user": "username",
    "host": "hostname1",
    "prefix": "/remote/calc/path1",
    "identity_file": "/path/to/private/key1",
    "paritions": [
      {
        "id": ["alias1", "alias2"],
        "name": "name_of_the_partition_in_the_queueing_system",
        "qos": "quality_of_service",
        "default": {
          "tasks_per_node": num_cpus,
          "exclusive": true,
          "nodes": 1,
          "interactive": false,
          "mem": "memory"
        },
        ...
      }
    ]
  },
  "remote2": {
    "user": "username",
    "host": "hostname2",
    "prefix": "/remote/calc/path2",
    "identity_file": "/path/to/private/key2",
    "paritions": []
  },
  ...,
  "local": {
    partitions: []
  } 
}
```

##### User specific part of the remote.json
There are to lines which have to be changes

```json5
{
  "mul-hpc": {
    "user": "username",
    "host": "hostname",
    "prefix": "/path/to/your/remote/calculation/folder",
    "identity_file": "/path/to/your/local/private_rsa_key_file",
    ...,
  "local" : {
    ...
  }
}
``` 

Please leave the rest of the file as it is `"local"`  referes always to the local machine.
As a reference my configuration looks like this

```json5
{
  "mul-hpc": {
    "user": "dnoeger",
    "host": "mul-hpc-81a.unileoben.ac.at",
    "prefix": "/calc/dnoeger",
    "identity_file": "/home/dominik/.ssh/id_rsa_mul_hpc",
    ...
}
```
  
   

