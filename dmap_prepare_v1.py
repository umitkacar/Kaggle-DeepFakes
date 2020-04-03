# Shortcut key : Comment (CTRL-K, CTRL-C) and Uncomment (CTRL-K, CTRL-U) are the same in Python and C++.
# python3  code.py > log.txt 2 > err.txt

import cv2
from mtcnn import MTCNN
import pandas as pd
import numpy as np
import os
import json
import sys
import tensorflow as tf

from skimage.morphology import convex_hull_image

MAIN_PATH = '/home/umit/xDataset/deepfake-detection-challenge/'
TRAIN_FOLDER = MAIN_PATH + 'train_full_videos/'
TRAIN_REAL_FACE_FOLDER = MAIN_PATH + 'train_real_face'
TRAIN_FAKE_FACE_FOLDER = MAIN_PATH + 'train_fake_face'
TRAIN_FAKE_DMAP_FOLDER = MAIN_PATH + 'train_fake_dmap'

IMG_FORMAT = '.png'

#width = 300
#height = 300
ex = 0
video_frame_jump_fake = 100
video_frame_jump_real = 100
video_frame_jump_original = 100

# load detector
detector = MTCNN()

# Train Main Path

for k in range(1):

    TRAIN_SUB_FOLDER = 'dfdc_train_part_' + str(k)
    print(f"Train samples: {len(os.listdir(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER)))}")

    train_list = list(os.listdir(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER)))
    ext_dict = []
    for file in train_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)
    print(f"Extensions: {ext_dict}")   

    for file_ext in ext_dict:
        print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    json_file = [file for file in train_list if  file.endswith('json')][0]
    print(f"JSON file: {json_file}")

    xMeta = pd.read_json(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,json_file),orient='index')

    for j in range(len(xMeta)):
        
        print('video =',xMeta.axes[0][j],'-',xMeta.label[j], '-',xMeta.original[j])

        if xMeta.label[j] == 'FAKE':
            
            #Read Orijinal video
            video_name_original= xMeta.original[j]
            video_path_original = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_original)
            video_name_split_original = video_name_original.split('.') 
            cap_original = cv2.VideoCapture(video_path_original)
            
             # Read FAKE video
            video_name_fake = xMeta.axes[0][j]
            video_path_fake = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_fake)
            video_name_split_fake = video_name_fake.split('.') 
            cap_fake = cv2.VideoCapture(video_path_fake)
            
            i = 0
            while(cap_original.isOpened() and cap_fake.isOpened()):

                print(str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame")

                ret_original, frame_original = cap_original.read()
                cap_original.set(cv2.CAP_PROP_POS_FRAMES, i)
                
                ret_fake, frame_fake = cap_fake.read()
                cap_fake.set(cv2.CAP_PROP_POS_FRAMES, i)
                
                i = i + video_frame_jump_original
        
                if (ret_original == True and ret_fake == True):
                    
                    # original video face detect
                    frame_temp_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB) 
                    face_original = detector.detect_faces(frame_temp_original)

                    if face_original: # empty control

                        for multi_face_original in range(len(face_original)):

                            box_original = face_original[multi_face_original]['box']
                            box_original = list(map(abs, box_original)) # ABS control
                            
                            x = box_original[0]
                            y = box_original[1]
                            w = box_original[2]
                            h = box_original[3]
                            
                            #cv2.rectangle(frame_original,(x,y),(x+w,y+h),(0,255,0),2)
                            crop_original = frame_original[y:y+h,x:x+w]
                            crop_fake = frame_fake[y:y+h,x:x+w]
                            
                            dmap = abs(crop_fake/1 - crop_original/1)
                            
                            #y, x = np.histogram(dmap_gray, bins=np.arange(255))
                            
                            # hist1,bins = np.histogram(dmap_gray[:,:,0].ravel(),256,[0,256])
                            # hist2,bins = np.histogram(dmap_gray[:,:,1].ravel(),256,[0,256])
                            # hist3,bins = np.histogram(dmap_gray[:,:,2].ravel(),256,[0,256])
                            
                            # print("hist1")
                            # print(hist1)
                            # print("hist2")
                            # print(hist2)
                            # print("hist3")
                            # print(hist3)
                            
                            dmap = np.where(dmap >= 20, 255, dmap)
                            dmap = np.where(dmap < 20, 0, dmap)
                            
                            dmap_orj = np.copy(dmap)
                            
                            try:
                                chull = convex_hull_image(dmap)
                                dmap[chull] = 255
                            except:
                                print("convex problem")
                            finally:
                                dmap = np.uint8(dmap)
                            
                            dmap = np.uint8(dmap)
                            dmap_orj = np.uint8(dmap_orj)
                                
                            # exp = np.expand_dims(dmap_gray,axis=2)
                            # dmap_gray = np.concatenate([exp,exp,exp],axis=2)
                            
                            
                            # dmap_gray = np.where(dmap_gray >= 10, 255, dmap_gray)
                            # dmap_gray = np.where(dmap_gray < 10, 0, dmap_gray)
                            # exp = np.expand_dims(dmap_gray,axis=2)
                            # dmap_gray = np.concatenate([exp,exp,exp],axis=2)
                            # dmap_gray = np.uint8(dmap_gray)

                            crop = np.concatenate([crop_original,crop_fake,dmap_orj,dmap],axis=1)
                            cv2.imshow("Crop original-fake frame window", crop)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                            # write original face detect
                            # img_name_original = video_name_split_original[0] + '_' + str(i) +  '_' + str(multi_face_original) + IMG_FORMAT
                            
                            # img_name_path_original = os.path.join(TRAIN_FAKE_FACE_FOLDER,img_name_original)
                            # cv2.imwrite(img_name_path_original, crop_original)
                    
                else:
                    break
                
            cap_original.release()
            cap_fake.release()
            cv2.destroyAllWindows()
            

        # if xMeta.label[j] == 'REAL':
        #     # Read REAL video
        #     video_name_real = xMeta.axes[0][j]
        #     video_path_real = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_real)
        #     video_name_split_real = video_name_real.split('.') 
        #     cap_real = cv2.VideoCapture(video_path_real)
            
        #     i = 0
        #     while(cap_real.isOpened()):

        #         print(str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame ")
                
        #         ret_real, frame_real = cap_real.read()
        #         cap_real.set(cv2.CAP_PROP_POS_FRAMES, i)
        #         i = i + video_frame_jump_real
                
        #         if (ret_real == True):
        #             # REAL video face detect
        #             frame_temp_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB) 
        #             face_real = detector.detect_faces(frame_temp_real)

        #             if face_real: # empty control

        #                 for multi_face_real in range(len(face_real)):
                        
        #                     box_real = face_real[multi_face_real]['box']
        #                     box_real = list(map(abs, box_real)) # ABS control
                            
        #                     x = box_real[0]
        #                     y = box_real[1]
        #                     w = box_real[2]
        #                     h = box_real[3]
                            
        #                     #cv2.rectangle(frame_real,(x,y),(x+w,y+h),(0,255,0),2)
        #                     crop_real = frame_real[y:y+h,x:x+w]
        #                     #cv2.imshow("Crop REAL Frame Window", crop_real)

        #                     # write REAL face detect
        #                     img_name_real = video_name_split_real[0] + '_' + str(i) + '_' + str(multi_face_real) + IMG_FORMAT
        #                     img_name_path_real = os.path.join(TRAIN_REAL_FACE_FOLDER,img_name_real)
        #                     cv2.imwrite(img_name_path_real, crop_real)

        #         else:
        #             break
                    
        #     cap_real.release()
        #     cv2.destroyAllWindows()     
# 11	aybumesmpk	0.162981	0.909046	0.60	23.862630	aybumesmpk.mp4	0	train	None	REAL
# 17	beboztfcme	0.448314	0.150025	0.45	41.950619	beboztfcme.mp4	0	train	None	REAL
# 18	bejhvclboh	0.326717	0.084289	0.40	54.149061	bejhvclboh.mp4	0	train	None	REAL
# 19	bffwsjxghk	0.191622	0.000640	0.35	188.073634	bffwsjxghk.mp4	0	train	None	REAL
# 23	bgwmmujlmc	0.342695	0.644088	0.65	89.553622	bgwmmujlmc.mp4	0	train	None	REAL
# 37	bwhlgysghg	0.376108	0.049915	0.40	109.975653	bwhlgysghg.mp4	0	train	None	REAL
# 38	bwipwzzxxu	0.473172	0.333369	0.55	323.809470	bwipwzzxxu.mp4	0	train	None	REAL
# 41	caifxvsozs	0.534639	0.160549	0.70	251.978801	caifxvsozs.mp4	0	train	None	REAL
# 49	cizlkenljw	0.077465	0.000337	0.30	31.477749	cizlkenljw.mp4	0	train	None	REAL
# 50	clrycekyst	0.275205	0.324562	0.40	168.693631	clrycekyst.mp4	0	train	None	REAL
# 51	cobjrlugvp	0.242216	0.040121	0.40	123.860438	cobjrlugvp.mp4	0	train	None	REAL
# 52	cppdvdejkc	0.236114	0.022023	0.40	29.403326	cppdvdejkc.mp4	0	train	None	REAL
# 68	ddepeddixj	0.395905	0.000476	0.40	31.872624	ddepeddixj.mp4	0	train	None	REAL
# 71	djxdyjopjd	0.284306	0.065937	0.40	208.258433	djxdyjopjd.mp4	0	train	None	REAL
# 85	ecujsjhscd	0.229562	0.616640	0.60	25.937014	ecujsjhscd.mp4	0	train	None	REAL
# 87	edyncaijwx	0.396912	0.472815	0.40	81.157620	edyncaijwx.mp4	0	train	None	REAL
# 90	efwfxwwlbw	0.120224	0.000335	0.30	42.166918	efwfxwwlbw.mp4	0	train	None	REAL
# 92	ehccixxzoe	1.054111	0.771815	0.80	22.696983	ehccixxzoe.mp4	0	train	None	REAL
# 95	ellavthztb	0.129231	0.000528	0.30	16.388932	ellavthztb.mp4	0	train	None	REAL
# 96	eqnoqyfquo	0.370194	0.165408	0.40	33.350365	eqnoqyfquo.mp4	0	train	None	REAL
# 97	erlvuvjsjf	0.077424	0.050213	0.30	27.941952	erlvuvjsjf.mp4	0	train	None	REAL
# 99	eudeqjhdfd	0.421151	0.496680	0.45	203.669805	eudeqjhdfd.mp4	0	train	None	REAL


# 0	aapnvogymq	0.736711	0.594691	0.80	67.820050	aapnvogymq.mp4	1	train	jdubbvfswz.mp4	FAKE
# 1	acifjvzvpm	0.699907	0.998962	0.90	150.660049	acifjvzvpm.mp4	1	train	kbvibjhfzo.mp4	FAKE
# 2	agqphdxmwt	1.079351	0.949184	0.90	174.422848	agqphdxmwt.mp4	1	train	aytzyidmgs.mp4	FAKE
# 3	altziddtxi	0.684690	0.486070	0.70	11.938622	altziddtxi.mp4	1	train	meawmsgiti.mp4	FAKE
# 4	apatcsqejh	0.374144	0.499256	0.40	57.475057	apatcsqejh.mp4	1	train	edyncaijwx.mp4	FAKE
# 5	apogckdfrz	1.347038	0.889846	0.80	96.195077	apogckdfrz.mp4	1	train	uonshkejav.mp4	FAKE
# 6	arlmiizoob	0.708348	0.346118	0.70	15.550470	arlmiizoob.mp4	1	train	meawmsgiti.mp4	FAKE
# 7	augtsuxpzc	1.187286	0.844676	0.80	65.325327	augtsuxpzc.mp4	1	train	ifbdbogiqn.mp4	FAKE
# 8	avvdgsennp	0.500000	0.500000	0.55	0.000000	avvdgsennp.mp4	1	train	gbqrgajyca.mp4	FAKE
# 9	axoygtekut	0.788844	0.978753	0.90	42.686201	axoygtekut.mp4	1	train	topyiohccg.mp4	FAKE
# 10	axwovszumc	0.652552	0.492618	0.70	35.097784	axwovszumc.mp4	1	train	rvoudrbyac.mp4	FAKE
# 11	aybumesmpk	0.162981	0.909046	0.60	23.862630	aybumesmpk.mp4	0	train	None	REAL
# 12	azpuxunqyo	0.736172	1.000303	0.90	17.998526	azpuxunqyo.mp4	1	train	jjyfvzxwwx.mp4	FAKE
# 13	bbhpvrmbse	1.008679	0.651970	0.80	149.996528	bbhpvrmbse.mp4	1	train	imzqmbfugn.mp4	FAKE
# 14	bbhtdfuqxq	0.463950	0.816920	0.75	67.749078	bbhtdfuqxq.mp4	1	train	cpjxareypw.mp4	FAKE
# 15	bchnbulevv	0.256049	0.999332	0.70	20.625421	bchnbulevv.mp4	1	train	iyefnuagav.mp4	FAKE
# 16	bctvsmddgq	0.501896	0.939994	0.90	54.327495	bctvsmddgq.mp4	1	train	ybjrqnqnno.mp4	FAKE
# 17	beboztfcme	0.448314	0.150025	0.45	41.950619	beboztfcme.mp4	0	train	None	REAL
# 18	bejhvclboh	0.326717	0.084289	0.40	54.149061	bejhvclboh.mp4	0	train	None	REAL
# 19	bffwsjxghk	0.191622	0.000640	0.35	188.073634	bffwsjxghk.mp4	0	train	None	REAL
# 20	bgaogsjehq	0.008226	0.450226	0.35	4.576257	bgaogsjehq.mp4	1	train	xzvrgckqkz.mp4	FAKE
# 21	bgmlwsoamc	0.469238	0.455347	0.55	143.330095	bgmlwsoamc.mp4	1	train	woshnzbxmc.mp4	FAKE
# 22	bguwlyazau	0.319031	0.561624	0.65	596.964875	bguwlyazau.mp4	1	train	znpdbbsfvj.mp4	FAKE
# 23	bgwmmujlmc	0.342695	0.644088	0.65	89.553622	bgwmmujlmc.mp4	0	train	None	REAL
# 24	blpchvmhxx	0.786929	0.998959	0.90	108.673042	blpchvmhxx.mp4	1	train	xqnykluhws.mp4	FAKE
# 25	bmbbkwmxqj	0.582906	0.451718	0.70	42.011475	bmbbkwmxqj.mp4	1	train	rrcsuwgpnd.mp4	FAKE
# 26	bmhvktyiwp	0.885171	1.000338	0.90	67.428697	bmhvktyiwp.mp4	1	train	gneufaypol.mp4	FAKE
# 27	bmioepcpsx	0.442822	0.651168	0.70	29.230862	bmioepcpsx.mp4	1	train	vmospzljws.mp4	FAKE
# 28	bntlodcfeg	0.817377	0.992031	0.90	117.663784	bntlodcfeg.mp4	1	train	ytufbmkdlq.mp4	FAKE
# 29	bopqhhalml	1.266281	1.001552	0.90	75.303492	bopqhhalml.mp4	1	train	oesxbvktem.mp4	FAKE
# 30	bourlmzsio	0.943624	0.999638	0.90	89.082478	bourlmzsio.mp4	1	train	bxzakyopjf.mp4	FAKE
# 31	brhalypwoo	0.415558	0.733585	0.70	87.866462	brhalypwoo.mp4	1	train	uuxqylnzls.mp4	FAKE
# 32	brvqtabyxj	1.254873	0.999054	0.90	60.220352	brvqtabyxj.mp4	1	train	ywvlvpvroj.mp4	FAKE
# 33	bsfmwclnqy	0.827212	0.999444	0.90	26.900938	bsfmwclnqy.mp4	1	train	ubplsigbvj.mp4	FAKE
# 34	bsqgziaylx	0.661591	0.784719	0.80	193.107422	bsqgziaylx.mp4	1	train	brwrlczjvi.mp4	FAKE
# 35	btiysiskpf	0.830468	0.999362	0.90	23.791576	btiysiskpf.mp4	1	train	gxhcuxulhi.mp4	FAKE
# 36	btjwbtsgln	0.936491	0.773379	0.80	154.787271	btjwbtsgln.mp4	1	train	xwcggrygwl.mp4	FAKE
# 37	bwhlgysghg	0.376108	0.049915	0.40	109.975653	bwhlgysghg.mp4	0	train	None	REAL
# 38	bwipwzzxxu	0.473172	0.333369	0.55	323.809470	bwipwzzxxu.mp4	0	train	None	REAL
# 39	byijojkdba	0.743780	0.999339	0.90	32.654583	byijojkdba.mp4	1	train	liniegczcx.mp4	FAKE
# 40	byyqectxqa	0.623622	0.684879	0.80	167.064911	byyqectxqa.mp4	1	train	fdcttsvjwf.mp4	FAKE
# 41	caifxvsozs	0.534639	0.160549	0.70	251.978801	caifxvsozs.mp4	0	train	None	REAL
# 42	cbbibzcoih	0.896132	0.991748	0.90	177.212322	cbbibzcoih.mp4	1	train	lietldeotq.mp4	FAKE
# 43	cbltdtxglo	1.186288	0.794726	0.80	287.565958	cbltdtxglo.mp4	1	train	svcnlasmeh.mp4	FAKE
# 44	cdaxixbosp	0.635876	0.979676	0.90	407.465348	cdaxixbosp.mp4	1	train	itzmdwutdu.mp4	FAKE
# 45	cdphtzqrvp	1.261565	0.999076	0.90	25.041494	cdphtzqrvp.mp4	1	train	ehtdtkmmli.mp4	FAKE
# 46	cffffbcywc	0.157233	0.817688	0.45	9.358773	cffffbcywc.mp4	1	train	ztbinwxgyu.mp4	FAKE
# 47	cglxirfaey	0.495914	0.983359	0.85	47.906867	cglxirfaey.mp4	1	train	qypgyrxcme.mp4	FAKE
# 48	cgvrgibpfo	0.431178	0.785719	0.70	38.577393	cgvrgibpfo.mp4	1	train	puppdcffcj.mp4	FAKE
# 49	cizlkenljw	0.077465	0.000337	0.30	31.477749	cizlkenljw.mp4	0	train	None	REAL

# 50	clrycekyst	0.275205	0.324562	0.40	168.693631	clrycekyst.mp4	0	train	None	REAL
# 51	cobjrlugvp	0.242216	0.040121	0.40	123.860438	cobjrlugvp.mp4	0	train	None	REAL
# 52	cppdvdejkc	0.236114	0.022023	0.40	29.403326	cppdvdejkc.mp4	0	train	None	REAL
# 53	crktehraph	0.770814	0.999968	0.90	86.106123	crktehraph.mp4	1	train	vrsinxahfh.mp4	FAKE
# 54	cthdnahrkh	0.772766	1.000761	0.90	88.018446	cthdnahrkh.mp4	1	train	vrsinxahfh.mp4	FAKE
# 55	cwbacdwrzo	0.831319	0.965982	0.90	158.330456	cwbacdwrzo.mp4	1	train	fecysfujzk.mp4	FAKE
# 56	cwqlvzefpg	0.500000	0.500000	0.55	0.000000	cwqlvzefpg.mp4	1	train	qeumxirsme.mp4	FAKE
# 57	cwsbspfzck	0.836355	0.881477	0.80	207.039497	cwsbspfzck.mp4	1	train	wtreibcmgm.mp4	FAKE
# 58	cwwandrkus	1.226464	0.999009	0.90	22.273276	cwwandrkus.mp4	1	train	kgbkktcjxf.mp4	FAKE
# 59	cxfujlvsuw	1.204763	0.998825	0.90	55.819052	cxfujlvsuw.mp4	1	train	qtnjyomzwo.mp4	FAKE
# 60	czfunozvwp	0.857695	0.751942	0.80	212.377721	czfunozvwp.mp4	1	train	caifxvsozs.mp4	FAKE
# 61	dafhtipaml	0.488607	0.949868	0.85	30.409952	dafhtipaml.mp4	1	train	kdodrvufdh.mp4	FAKE
# 62	dbhoxkblzx	1.107339	0.998786	0.90	63.554115	dbhoxkblzx.mp4	1	train	qtnjyomzwo.mp4	FAKE
# 63	dbhrpizyeq	0.712014	0.999245	0.90	34.900219	dbhrpizyeq.mp4	1	train	ffcwhpnpuw.mp4	FAKE
# 64	dboxtiehng	0.733501	0.999731	0.90	22.247769	dboxtiehng.mp4	1	train	jjyfvzxwwx.mp4	FAKE
# 65	dbzcqmxzaj	1.075194	0.998889	0.90	25.422758	dbzcqmxzaj.mp4	1	train	yxyhvdlrgk.mp4	FAKE
# 66	dcamvmuors	0.690832	0.977553	0.90	204.310760	dcamvmuors.mp4	1	train	iuzdfwsefw.mp4	FAKE
# 67	dcuiiorugd	0.611972	0.812256	0.80	103.509452	dcuiiorugd.mp4	1	train	wapebjxejr.mp4	FAKE
# 68	ddepeddixj	0.395905	0.000476	0.40	31.872624	ddepeddixj.mp4	0	train	None	REAL
# 69	degpbqvcay	0.498783	0.664268	0.75	16.667535	degpbqvcay.mp4	1	train	ptokilxwcx.mp4	FAKE
# 70	dhjmzhrcav	0.824399	1.001186	0.90	385.534810	dhjmzhrcav.mp4	1	train	tdohqkzvbk.mp4	FAKE
# 71	djxdyjopjd	0.284306	0.065937	0.40	208.258433	djxdyjopjd.mp4	0	train	None	REAL
# 72	dkwjwbwgey	0.463009	0.960803	0.85	28.803570	dkwjwbwgey.mp4	1	train	rfzzrftgco.mp4	FAKE
# 73	dofusvhnib	0.380121	0.645692	0.65	15.042481	dofusvhnib.mp4	1	train	xnfwdpptym.mp4	FAKE
# 74	dptrzdvwpg	1.073473	0.436530	0.70	53.092186	dptrzdvwpg.mp4	1	train	iiomvouemm.mp4	FAKE
# 75	dqnyszdong	0.500000	0.500000	0.55	0.000000	dqnyszdong.mp4	1	train	qeumxirsme.mp4	FAKE
# 76	dqswpjoepo	0.335944	0.684415	0.65	140.729755	dqswpjoepo.mp4	1	train	kydlpqfrvv.mp4	FAKE
# 77	drgjzlxzxj	0.584105	0.925291	0.90	45.506420	drgjzlxzxj.mp4	1	train	yagllixjvh.mp4	FAKE
# 78	drtbksnpol	0.001416	0.633634	0.55	4.407734	drtbksnpol.mp4	1	train	dzyuwjkjui.mp4	FAKE
# 79	dtocdfbwca	0.744710	0.936769	0.90	306.879200	dtocdfbwca.mp4	1	train	dakiztgtnw.mp4	FAKE
# 80	dvakowbgbt	0.883812	0.768940	0.80	315.802952	dvakowbgbt.mp4	1	train	slwkmefgde.mp4	FAKE
# 81	dzieklokdr	0.726805	0.647659	0.80	39.104926	dzieklokdr.mp4	1	train	pylnolwenx.mp4	FAKE
# 82	dzqwgqewhu	0.756197	0.999382	0.90	18.017744	dzqwgqewhu.mp4	1	train	xwcggrygwl.mp4	FAKE
# 83	dzvyfiarrq	0.479813	0.819962	0.75	54.428875	dzvyfiarrq.mp4	1	train	bgwmmujlmc.mp4	FAKE
# 84	eahlqmfvtj	0.003894	0.654524	0.55	6.635166	eahlqmfvtj.mp4	1	train	lyvlnqduqg.mp4	FAKE
# 85	ecujsjhscd	0.229562	0.616640	0.60	25.937014	ecujsjhscd.mp4	0	train	None	REAL
# 86	ecuvtoltue	0.725336	0.999389	0.90	37.651805	ecuvtoltue.mp4	1	train	xngpzquyhs.mp4	FAKE
# 87	edyncaijwx	0.396912	0.472815	0.40	81.157620	edyncaijwx.mp4	0	train	None	REAL
# 88	eekozbeafq	0.445232	0.746758	0.70	29.111719	eekozbeafq.mp4	1	train	olakcrnuro.mp4	FAKE
# 89	eeyhxisdfh	0.000954	0.168466	0.35	7.518884	eeyhxisdfh.mp4	1	train	lyvlnqduqg.mp4	FAKE
# 90	efwfxwwlbw	0.120224	0.000335	0.30	42.166918	efwfxwwlbw.mp4	0	train	None	REAL
# 91	ehbnclaukr	0.021992	0.759455	0.55	7.048663	ehbnclaukr.mp4	1	train	gipbyjfxfp.mp4	FAKE
# 92	ehccixxzoe	1.054111	0.771815	0.80	22.696983	ehccixxzoe.mp4	0	train	None	REAL
# 93	ehdkmxgtxh	0.574853	1.000588	0.90	54.319668	ehdkmxgtxh.mp4	1	train	bejhvclboh.mp4	FAKE
# 94	elginszwtk	0.029378	0.878337	0.55	6.898271	elginszwtk.mp4	1	train	gipbyjfxfp.mp4	FAKE
# 95	ellavthztb	0.129231	0.000528	0.30	16.388932	ellavthztb.mp4	0	train	None	REAL
# 96	eqnoqyfquo	0.370194	0.165408	0.40	33.350365	eqnoqyfquo.mp4	0	train	None	REAL
# 97	erlvuvjsjf	0.077424	0.050213	0.30	27.941952	erlvuvjsjf.mp4	0	train	None	REAL
# 98	etmcruaihe	1.196796	0.849543	0.80	133.241890	etmcruaihe.mp4	1	train	afoovlsmtx.mp4	FAKE
# 99	eudeqjhdfd	0.421151	0.496680	0.45	203.669805	eudeqjhdfd.mp4	0	train	None	REAL



# file	dmap	cls	model_label	l_var	filename	label	split	original	real_label
# 6	anpuvshzoo	0.461521	0.324275	0.55	85.670375	anpuvshzoo.mp4	0	train	None	REAL
# 10	atkdltyyen	0.150056	0.311172	0.35	236.917158	atkdltyyen.mp4	0	train	None	REAL
# 16	bgvhtpzknn	0.248439	0.344296	0.40	14.211199	bgvhtpzknn.mp4	0	train	None	REAL
# 19	bilnggbxgu	0.303040	0.338753	0.45	13.859492	bilnggbxgu.mp4	0	train	None	REAL
# 34	ciyoudyhly	0.382069	0.141857	0.45	83.270807	ciyoudyhly.mp4	0	train	None	REAL
# 35	cmbzllswnl	0.298775	0.431670	0.48	169.532088	cmbzllswnl.mp4	0	train	None	REAL
# 43	dsjbknkujw	0.509386	0.290796	0.60	697.769649	dsjbknkujw.mp4	0	train	None	REAL

# 	file	dmap	cls	model_label	l_var	filename	label	split	original	real_label
# 0	aagfhgtpmv	8.315495e-01	0.447959	0.70	132.765575	aagfhgtpmv.mp4	1	train	vudstovrck.mp4	FAKE
# 1	adhsbajydo	5.000000e-01	0.500000	0.55	0.000000	adhsbajydo.mp4	1	train	fysyrqfguw.mp4	FAKE
# 2	ahfazfbntc	8.710228e-01	0.999991	0.85	61.660659	ahfazfbntc.mp4	1	train	sunqwnmlkx.mp4	FAKE
# 3	aipfdnwpoo	6.841227e-01	0.424647	0.65	40.552996	aipfdnwpoo.mp4	1	train	ygdgwyqyut.mp4	FAKE
# 4	aklqzsddfl	3.893651e-02	0.953220	0.50	7.896317	aklqzsddfl.mp4	1	train	lyvlnqduqg.mp4	FAKE
# 5	alninxcyhg	1.171070e+00	0.944424	0.85	88.078968	alninxcyhg.mp4	1	train	tqhbgzfwsf.mp4	FAKE
# 7	aqpnvjhuzw	5.649498e-01	0.935552	0.75	45.353306	aqpnvjhuzw.mp4	1	train	xngpzquyhs.mp4	FAKE
# 8	arlmiizoob	8.473433e-01	0.577286	0.75	10.160336	arlmiizoob.mp4	1	train	meawmsgiti.mp4	FAKE
# 9	asdpeebotb	1.067078e-01	0.954604	0.45	7.588890	asdpeebotb.mp4	1	train	znjupdqnwo.mp4	FAKE
# 11	avtycwsgyb	8.728239e-08	0.753430	0.45	7.052954	avtycwsgyb.mp4	1	train	qzklcjjxdq.mp4	FAKE
# 12	axwgcsyphv	5.000000e-01	0.500000	0.55	0.000000	axwgcsyphv.mp4	1	train	mfnowqfdwl.mp4	FAKE
# 13	azsmewqghg	5.604340e-01	0.996496	0.75	115.001420	azsmewqghg.mp4	1	train	djxdyjopjd.mp4	FAKE
# 14	bchnbulevv	2.863994e-01	0.891566	0.55	20.387381	bchnbulevv.mp4	1	train	iyefnuagav.mp4	FAKE
# 15	bgaogsjehq	1.514243e-02	0.383947	0.30	4.549132	bgaogsjehq.mp4	1	train	xzvrgckqkz.mp4	FAKE
# 17	bhaaboftbc	3.035129e-01	0.676413	0.52	17.642286	bhaaboftbc.mp4	1	train	rlvgtsjyer.mp4	FAKE
# 18	bhsluedavd	4.051495e-01	0.843349	0.65	156.246601	bhsluedavd.mp4	1	train	kydlpqfrvv.mp4	FAKE
# 20	bmjmjmbglm	4.717565e-01	0.808372	0.70	12.926624	bmjmjmbglm.mp4	1	train	mmhqllmlew.mp4	FAKE
# 21	bnjcdrfuov	3.859063e-01	0.422739	0.45	15.898820	bnjcdrfuov.mp4	1	train	ellavthztb.mp4	FAKE
# 22	bofqajtwve	5.073280e-01	1.000126	0.75	23.349892	bofqajtwve.mp4	1	train	dbtbbhakdv.mp4	FAKE
# 23	bourlmzsio	9.477384e-01	0.999774	0.85	90.072824	bourlmzsio.mp4	1	train	bxzakyopjf.mp4	FAKE
# 24	bqdjzqhcft	8.702849e-01	1.000007	0.85	204.414138	bqdjzqhcft.mp4	1	train	ytufbmkdlq.mp4	FAKE
# 25	bqkdbcqjvb	5.000000e-01	0.500000	0.55	0.000000	bqkdbcqjvb.mp4	1	train	atvmxvwyns.mp4	FAKE
# 26	bqtuuwzdtr	5.782236e-01	0.925686	0.75	58.681209	bqtuuwzdtr.mp4	1	train	yagllixjvh.mp4	FAKE
# 27	btjwbtsgln	9.490252e-01	0.692144	0.75	151.979222	btjwbtsgln.mp4	1	train	xwcggrygwl.mp4	FAKE
# 28	btxlttbpkj	6.193620e-01	0.990393	0.80	76.298056	btxlttbpkj.mp4	1	train	fhghkqdkhe.mp4	FAKE
# 29	bwuwstvsbw	6.834639e-01	0.999610	0.80	7.967237	bwuwstvsbw.mp4	1	train	xzvrgckqkz.mp4	FAKE
# 30	byfenovjnf	5.424230e-04	0.497873	0.45	6.945678	byfenovjnf.mp4	1	train	gipbyjfxfp.mp4	FAKE
# 31	byyqectxqa	5.819409e-01	0.696921	0.65	192.684014	byyqectxqa.mp4	1	train	fdcttsvjwf.mp4	FAKE
# 32	bzmdrafeex	6.010718e-01	0.619552	0.70	175.419465	bzmdrafeex.mp4	1	train	sqwvfgwdxr.mp4	FAKE
# 33	cgvrgibpfo	4.710168e-01	0.860362	0.70	38.345288	cgvrgibpfo.mp4	1	train	puppdcffcj.mp4	FAKE
# 36	cqhngvpgyi	9.268206e-01	0.947209	0.85	128.245277	cqhngvpgyi.mp4	1	train	tfoixxmpoo.mp4	FAKE
# 37	cuzrgrbvil	8.096527e-01	0.244012	0.70	24.497253	cuzrgrbvil.mp4	1	train	iiomvouemm.mp4	FAKE
# 38	cycacemkmt	5.000000e-01	0.500000	0.55	0.000000	cycacemkmt.mp4	1	train	atvmxvwyns.mp4	FAKE
# 39	ddqccgmtka	7.644740e-01	0.870067	0.80	25.691264	ddqccgmtka.mp4	1	train	qokxxuayqn.mp4	FAKE
# 40	diuzrpqjli	9.884929e-01	0.471742	0.70	76.658128	diuzrpqjli.mp4	1	train	smggzgxymo.mp4	FAKE
# 41	dnexlwbcxq	1.353356e+00	0.999735	0.85	43.059629	dnexlwbcxq.mp4	1	train	ezaajaswoe.mp4	FAKE
# 42	dofusvhnib	3.591324e-01	0.701379	0.52	15.684084	dofusvhnib.mp4	1	train	xnfwdpptym.mp4	FAKE
# 44	dzvyfiarrq	1.737519e-01	0.476705	0.45	48.363642	dzvyfiarrq.mp4	1	train	bgwmmujlmc.mp4	FAKE
# 45	ebebgmtlcu	1.378463e+00	0.999431	0.85	46.480297	ebebgmtlcu.mp4	1	train	iufotyxgzb.mp4	FAKE
# 46	ekhacizpah	5.247387e-01	0.536867	0.65	142.517238	ekhacizpah.mp4	1	train	egghxjjmfg.mp4	FAKE
# 47	epymyyiblu	3.175858e-02	0.424280	0.45	492.435222	epymyyiblu.mp4	1	train	svcnlasmeh.mp4	FAKE
# 48	etdcqxabww	2.221442e-02	0.479123	0.45	6.752075	etdcqxabww.mp4	1	train	gipbyjfxfp.mp4	FAKE
# 49	etohcvnzbj	1.296046e+00	0.497080	0.70	247.358033	etohcvnzbj.mp4	1	train	bdnaqemxmr.mp4	FAKE
