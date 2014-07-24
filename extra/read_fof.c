/* interface to P-FoF 
   GM Nov 2005 */
/* still to implement:
   -extraction of one/all groups from the corresponding snapshot
   -extraction of all particles from the CM of a given group/all groups 
*/

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define BASEDIR "/home/lbignone/simulations/MARE_230_D"

struct FofData {
  int *GroupLen;   /* total number of particles */
  int *GroupLenType; /* number of particles per type (6 int per group) */			
  double *GroupMassType; /* total mass of group per type (d double per group) */
  float *GroupCM; /* group center of mass (3 float per group) */
  float *GroupSfr; /* group sfr (KLAUS, THIS IS IDENTICALLY NOUGHT!) */
  int *GroupOffset; /* offset in byte to access the ID list of the group in the id file */
  int *GroupIDs; /* list of all the Gadget Ids of all groups in task */
};


struct FoF {
  int Ntasks, TotNgroups;
  int *Nids,*Ngroups;
  struct FofData *fof;
} d;

int iformat;

main(int argc, char *argv[])
{
  int sn, i, j, k;
  int iT;
  float m, mM;
  

  void fof_load_catalogue(int, int), ListGroups();
  void ListOne(int, int);
  int fof_load_catalogue_head(int);


  if(argc==1) {
    printf("Arguments: snapshot_number [-l] [-m] [-M] [-x] [-g nTask idxTask]\n");
    printf("    -l stats of all group, all details\n");
    printf("    -m stats of most massive group per task, all details\n");
    printf("    -M stats of most massive group, all details\n");
    printf("    -x lists all group, format Idx_tot Npart_dm Mass_dm X Y X nTask idxTask\n");
    printf("    -g nTask idxTask stats of the group, all details\n");
    exit(-1);
  }


  sn = atoi(argv[1]);
  d.Ntasks=fof_load_catalogue_head(sn);
  d.Nids= malloc( d.Ntasks * sizeof(int));
  d.Ngroups= malloc( d.Ntasks * sizeof(int));
  d.fof = malloc( d.Ntasks * sizeof(struct FofData));

  printf("P-Fof run on %d nodes and find %d groups in total.\n\n",d.Ntasks,d.TotNgroups);

  for(i=0;i<d.Ntasks; i++) {
    fof_load_catalogue(sn,i);
    printf(" Task %d %d groups %d ids\n",i,d.Ngroups[i],d.Nids[i]);
  }

  iformat=0;
  i=2;
  while(i<argc) {
    if(!strcmp(argv[i],"-l")) { /* list all */
      i++;
      ListGroups();
    }
    else if(!strcmp(argv[i],"-x")) { /* list all, formatted */
      i++; iformat=1;
      ListGroups();
      iformat=0;
    }
    else if(!strcmp(argv[i],"-m")) { /* list first per tasks */
      i++;
      for(j=0; j<d.Ntasks; j++)
	ListOne(j,0);
    }
    else if(!strcmp(argv[i],"-g")) { /* list given group */
      i++;
      iT = atoi(argv[i]); i++;
      j = atoi(argv[i]); i++;
      ListOne(iT,j); //????
    }
    else if(!strcmp(argv[i],"-M")) { /* list most massive at all */
      i++;
      iT=0;       
      mM=0.0;
      for(k=0;k<6;k++)	mM += d.fof[0].GroupMassType[ 6*k ];
	  for(j=1; j<d.Ntasks; j++) {
	    m=0.0;
	    for(k=0;k<6;k++)   m += d.fof[j].GroupMassType[ 6*k ];
	    if(m > mM) {
	      iT=j;
	      mM=m;
	    }
	  }
	  ListOne(iT,0);
    }
    i++;
  }


  /* here fof data are at your disposal. Access them in order of task
     (d.fof[nTask].*)
     d.Ngroups[nTask] is the number of groups each task identified
     d.Nids[nTask] is the total number of particles considered by that task
     for each task you have all the characteristics of all groups fof found 
     (d.fof[nTask].GroupLen[ 0,1,2..d.Ngroups[nTask] ] total number
     of particles, etc) */
}

void fof_load_catalogue(int num, int Task)
{
  FILE *fd;
  char buf[500];
  int Ngroups, Nids, TotNgroups, NTask;
  int *GroupLen, *GroupOffset, *GroupLenType;
  double *GroupMassType;
  float *GroupCM, *GroupSfr;
  int *GroupIDs;

  sprintf(buf, "%s/groups_%03d/%s_%03d.%d", BASEDIR, num, "group_tab", num, Task);
  if(!(fd = fopen(buf, "r")))
    {
      printf("can't open file `%s`\n", buf);
      exit(-1);
    }

  fread(&Ngroups, sizeof(int), 1, fd);
  fread(&Nids, sizeof(int), 1, fd);
  d.Nids[Task]=Nids;
  d.Ngroups[Task]=Ngroups;

  d.fof[Task].GroupLen = malloc(1 + Ngroups * sizeof(int));   
  d.fof[Task].GroupLenType = malloc(6 * (1 + Ngroups * sizeof(int)));
  d.fof[Task].GroupMassType = malloc(6 * (1 + Ngroups * sizeof(double)));
  d.fof[Task].GroupCM = malloc(3 * (1 + Ngroups * sizeof(float)));
  d.fof[Task].GroupSfr = malloc((1 + Ngroups * sizeof(float)));
  d.fof[Task].GroupOffset = malloc(1 + Ngroups * sizeof(int));
  d.fof[Task].GroupIDs = malloc(1 + Nids * sizeof(int));



  fread(&TotNgroups, sizeof(int), 1, fd);
  fread(&NTask, sizeof(int), 1, fd);
  fread(d.fof[Task].GroupLen, sizeof(int), Ngroups, fd);
  fread(d.fof[Task].GroupOffset, sizeof(int), Ngroups, fd);

  fread(d.fof[Task].GroupLenType, 6 * sizeof(int), Ngroups, fd);
  fread(d.fof[Task].GroupMassType, 6 * sizeof(double), Ngroups, fd);
  fread(d.fof[Task].GroupCM, 3 * sizeof(float), Ngroups, fd);
  fread(d.fof[Task].GroupSfr, sizeof(float), Ngroups, fd);
  fclose(fd);
  sprintf(buf, "%s/groups_%03d/%s_%03d.%d", BASEDIR, num, "group_ids", num, Task);
  printf("Reading file %s...\n", buf);
  if(!(fd = fopen(buf, "r")))
    {
      printf("can't open file `%s`\n", buf);
      exit(-2);
    }

  fread(&Ngroups, sizeof(int), 1, fd);
  fread(&Nids, sizeof(int), 1, fd);
  fread(&TotNgroups, sizeof(int), 1, fd);
  fread(&NTask, sizeof(int), 1, fd);
  fread(d.fof[Task].GroupIDs, sizeof(int), Nids, fd);
  fclose(fd);



}

int fof_load_catalogue_head(int num)
{
  FILE *fd;
  char buf[500];
  int Ngroups, Nids, TotNgroups,NTask;


  sprintf(buf, "%s/groups_%03d/%s_%03d.%d", BASEDIR, num, "group_tab", num, 0);
  printf("File: %s\n",buf);
  if(!(fd = fopen(buf, "r")))
    {
      printf("can't open file `%s`\n", buf);
      exit(-1);
    }

  fread(&Ngroups, sizeof(int), 1, fd);
  fread(&Nids, sizeof(int), 1, fd);
  fread(&TotNgroups, sizeof(int), 1, fd);
  fread(&NTask, sizeof(int), 1, fd);

  fclose(fd);

  d.TotNgroups=TotNgroups;
  return NTask;
}


void ListGroups()
{
  int i,j,k,n;

  n=1;
  for(i=0;i<d.Ntasks; i++) 
    for(j=0;j<d.Ngroups[i];j++) {
      if(iformat==0) {
	printf(" Group %d/%d (%d) Npart %d\n",j,i,n,d.fof[i].GroupLen[j]);
	printf("    center of mass in : %f %f %f\n",
	       d.fof[i].GroupCM[ (3*j) ],
	       d.fof[i].GroupCM[ (3*j) +1],  d.fof[i].GroupCM[ (3*j) +2]);
	printf("    particle per type: \n");
	for(k=0; k<6; k++)
	  printf("          type %d: %d\n",k,d.fof[i].GroupLenType[ 6*j + k ]);
	printf("    particle mass per type: \n");
	for(k=0; k<6; k++)
	  printf("          type %d: %lf\n",k,d.fof[i].GroupMassType[ 6*j + k ]);
	printf("    group sfr: %f\n",d.fof[i].GroupSfr[j]);
	printf("    group offset: %d\n",d.fof[i].GroupOffset[j]);
      } else {
	printf("%d %d %f    %f %f %f   %d %d\n",n, d.fof[i].GroupLenType[ 6*j+1], 
	       d.fof[i].GroupMassType[ 6*j + 1 ], d.fof[i].GroupCM[ (3*j) ],
	       d.fof[i].GroupCM[ (3*j) +1],  d.fof[i].GroupCM[ (3*j) +2], i, j);
      }
      n++;
    }
  
}

void ListOne(int i, int j)
{
  int k;

      printf(" Group %d/%d Npart %d\n",j,i,d.fof[i].GroupLen[j]);
      printf("    center of mass in : %f %f %f\n",
      	     d.fof[i].GroupCM[ (3*j) ],
      	     d.fof[i].GroupCM[ (3*j) +1],  d.fof[i].GroupCM[ (3*j) +2]);
      printf("    particle per type: \n");
      for(k=0; k<6; k++)
	printf("          type %d: %d\n",k,d.fof[i].GroupLenType[ 6*j + k ]);
      printf("    particle mass per type: \n");
      for(k=0; k<6; k++)
	printf("          type %d: %lf\n",k,d.fof[i].GroupMassType[ 6*j + k ]);
      printf("    group sfr: %f\n",d.fof[i].GroupSfr[j]);
      printf("    group offset: %d\n",d.fof[i].GroupOffset[j]);

  
}

