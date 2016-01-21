% example from Ermentrout in Arbib 2nd edition.
while (1==1)

	w = [-0.424730,  0.243325, -0.267939,  0.308063, -0.0370201,  0.3944969;
    	 -0.166832,  0.474204, -0.0151443, 0.476774,  0.211162,   0.401305;
		 -0.427914,  0.370044, -0.0675567, 0.276535,  0.45988 ,  -0.457168;
		 -0.362131,  0.033334, -0.196538, -0.037606, -0.125548,   0.143851;
    	  0.429334, -0.306886,  0.402954, -0.166799, -0.45518,   -0.0304156;
		  0.294663, -0.346348, -0.138444,  0.334973,  0.13884,   -0.364227];

	% or try other random values 
	%mu is the gain;
	% interesting between 4 and 8 in particular, and 15 and 19


	mu=6;	  
	dums='';
	mu=str2num(input('New mu value ','s'))


	%%variable symmetry sym=0 .. 1
	sym = 0.;
	w = (1-sym/2)*w + sym/2*w';

	w = w * mu;

	u  = 0.1*ones(6,1);	

	dt = 0.01;
	tau= 1;

	endtime =150;
	nt=endtime/dt;
	ply=zeros(6,nt+1);
	plx=[0:dt:endtime];

	for it=1:nt
		u= (1-dt/tau)*u + dt/tau*tanh(w*u);
		ply(:,it) =  u;
	end	
plot(plx,ply');
end
