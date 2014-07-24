pro read_fof,snr,d,d_id=d_id,swap_endian=swap_endian,ext=ext

ext = '.1'
;IF NOT(IS_DEF(ext)) then ext='.0'


dir ='/nfs/data-mpa/cecilia/ZoomICs/output-GA1-feed-0.5-Er-0.3/'

 openr,1,dir+'./groups_'+snr+'/group_tab_'+snr+ext,swap_endian=swap_endian
print,'dir=',dir+'./groups_'+snr+'/group_tab_'+snr+ext


   Ngroups=0L
   Nids=0L
   TotNgroups=0L
   NTask=0L
   readu,1,Ngroups,Nids,TotNgroups,NTask
   print,Ngroups,Nids,TotNgroups,NTask
   GroupLen=lonarr(Ngroups)
   readu,1,GroupLen
   GroupOffset=lonarr(Ngroups)
   readu,1,GroupOffset
   GroupLenType=lonarr(6,Ngroups)
   readu,1,GroupLenType
   GroupMassType=dblarr(6,Ngroups)
   readu,1,GroupMassType
   GroupCM=fltarr(3,Ngroups)
   readu,1,GroupCM
   GroupSfr=fltarr(Ngroups)
   readu,1,GroupSfr
   close,1

   d = { fof , Ngroups:Ngroups,$
               Nids:Nids,$
               TotNgroups:TotNgroups,$
               NTask:NTask,$
               GroupLen:GroupLen,$
               GroupOffset:GroupOffset,$
               GroupLenType:GroupLenType,$
               GroupMassType:GroupMassType,$
               GroupCM:GroupCM,$
               GroupSfr:GroupSfr}

   openr,1,dir+'./groups_'+snr+'/group_ids_'+snr+ext,swap_endian=swap_endian
 
   readu,1,Ngroups,Nids,TotNgroups,NTask

   groupIDs=ulonarr(Nids)
   readu,1,groupIDs

   close,1

   d_id = { fof_id , Ngroups:Ngroups,$
                     Nids:Nids,$
                     TotNgroups:TotNgroups,$
                     NTask:NTask,$
	             groupIDs:groupIDs}
 

end

pro test_fof
setcolors
   ihal=0
   readnew,'snap_092',h,'HEAD'
   readnew,'snap_092',pos_t,'POS'
   readnew,'snap_092',id_t,'ID'

   pos=pos_t
   idns=h.npart(0)+h.npart(1)+h.npart(2)+h.npart(3)
   id=id_t
   ii=SORT(id_t(0:idns-1))
   pos(0,0:idns-1)=pos_t(0,ii)
   pos(1,0:idns-1)=pos_t(1,ii)
   pos(2,0:idns-1)=pos_t(2,ii)

   read_fof,'092',d,d_id=d_id,ext='.0'

   print,d.GroupMassType(*,d.GroupOffset(ihal))*1e10
   print,d.GroupCM(*,d.GroupOffset(ihal))

   x=fltarr(d.GroupLen(ihal))
   y=fltarr(d.GroupLen(ihal))
   z=fltarr(d.GroupLen(ihal))

   FOR j=0L,d.GroupLen(ihal)-1 DO BEGIN
      myid=d_id.groupIDs(d.GroupOffset(ihal)+j)
      IF myid LT idns THEN BEGIN
         x(j)=pos(0,myid-1)
         y(j)=pos(1,myid-1)
         z(j)=pos(2,myid-1)
      END ELSE BEGIN
         jj=WHERE(id EQ myid)
         x(j)=pos(0,jj(0))
         y(j)=pos(1,jj(0))
         z(j)=pos(2,jj(0))
      END
   END


   plot,x,y,psym=3,xst=1,yst=1
   circle,d.GroupCM(0,d.GroupOffset(ihal)),d.GroupCM(1,d.GroupOffset(ihal)),1000,xc,yc
   oplot,xc,yc,col=mycolor(4)

end


FUNCTION IS_DEF, x
aux = SIZE(x)
RETURN, aux(N_ELEMENTS(aux)-2) NE 0
END
