import taichi as ti
import numpy as np
import h5py
import time
import taichi.math as tm
import os
import torch
import math

# a self_consistent mpm solver, i.e. shall input samples_all and output samples_all
@ti.data_oriented
class MPM_solver:
    def __init__(self, dim, n_particles):
        self.initialize(dim, n_particles)

    def initialize(self, dim, n_particles):
        self.dim = dim
        # By default y-axis is vertical (gravity)-- this is for a convenient 2D visualization
        DTYPE = self.dtype = ti.f32

        ## popular setting for unknown reason, subject to change
        quality = 1
        self.n_particles = n_particles
        n_grid = self.n_grid = int(100 * quality)
        self.dx, self.inv_dx = 1 / n_grid, float(n_grid)

        # particle volume subject to change!!!!
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho

        self.gravity = ti.Vector.field(self.dim, dtype=DTYPE, shape=()) # gravity
        self.mu =  ti.field(dtype=DTYPE, shape=n_particles)
        self.lam = ti.field(dtype=DTYPE, shape=n_particles) # material parameter for each particle, not intialized yet
        # store those physical quantities at each timestep
        self.x = ti.Vector.field(dim, dtype=DTYPE, shape=(n_particles)) # position
        self.x_2d = ti.Vector.field(2, dtype=DTYPE, shape=(n_particles)) # 2d for quick visualization
        self.x_initial = ti.Vector.field(dim, dtype=DTYPE, shape=(n_particles)) # initial position, for saving data
        self.u = ti.Vector.field(dim, dtype=DTYPE, shape=(n_particles)) # displacement, = x - x_initial
        self.v = ti.Vector.field(dim, dtype=DTYPE, shape=(n_particles)) # velocity
        self.F = ti.Matrix.field(dim, dim, dtype=DTYPE, shape=(n_particles)) # deformation gradient
        self.F_disp = ti.Matrix.field(dim, dim, dtype=DTYPE, shape=(n_particles)) # deformation gradient
        self.yield_stress = ti.field(dtype=DTYPE, shape=n_particles) # plasticity yield threshold
        self.material = ti.field(dtype=int, shape=n_particles) # material id, for now assume only one material
        self.particle_mass = ti.field(dtype=DTYPE, shape=n_particles)

        self.res = res = (n_grid, n_grid) if dim == 2 else (n_grid, n_grid, n_grid)
        self.grid_v_in = ti.Vector.field(dim, dtype=DTYPE, shape=res) # grid node momentum/velocity
        self.grid_v_out = ti.Vector.field(dim, dtype=DTYPE, shape=res) # grid node momentum/velocity after grid update
        self.grid_m = ti.field(dtype=DTYPE, shape=res) # grid node mass
        self.time = 0.0
        self.padding = 3
        self.grid_postprocess = []
        self.grid_postprocess_lead = []
        self.BC_anchor_position = []
        self.BC_anchor_velocity = []


    def initialize_parameters(self):
        self.gravity[None] = [0, -1.0, 0]
        self.gravitational_accelaration = 0

        # material
        E, nu = 0.05 * 10, 0.35 # two scalars
        self._mu, self._lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.use_plasticity_vm = False
        self._yield_stress = 10

        self.mu.fill(self._mu)
        self.lam.fill(self._lam)
        self.yield_stress.fill(self._yield_stress)


    def step0(self, sample_x, sample_X, sample_v, sample_F, sample_mass, omit_xinit_and_mass = False):
        s1 = time.time()
        self.initialize_parameters()
        s2 = time.time()
        self.load_from_nn(sample_x, sample_X, sample_v, sample_F, sample_mass, omit_xinit_and_mass)
        s3 = time.time()
        print("Time for reduced_mpm_solver - initialize_parameters(): ", s2 - s1)
        print("Time for reduced_mpm_solver - initialize_variables(..): ", s3 - s2)

    def load_from_nn(self, sample_x, sample_X, sample_v, sample_F, sample_mass, omit_xinit_and_mass = False):
        # input_tensor is (n_particles, dim)
        # sample_F is of shape (n, 9)
        sample_F = torch.reshape(sample_F, (-1,3,3)) # arranged by rowmajor

        self.x.from_numpy(sample_x)
        self.v.from_numpy(sample_v)
        self.F.from_numpy(sample_F)
        self.F_disp.from_numpy(sample_F)

        if not omit_xinit_and_mass:
            self.x_initial.from_numpy(sample_X)
            self.particle_mass.from_numpy(sample_mass)

        use_displacement = True
        # need to be set, F and mass
        for i in range(self.n_particles):
            if use_displacement:
                self.F[i] += ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.material[i] = 0

    @ti.kernel
    def update_x_2d(self):
        for p in range(0, self.n_particles):
            self.x_2d[p][0] = self.x[p][0]
            self.x_2d[p][1] = self.x[p][1]

    def load_from_sampling(self, sampling_h5, flag_p2d = True):
        if not os.path.exists(os.getcwd() + sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(os.getcwd() + sampling_h5, 'r')
        x, q, particle_mass, f_tensor = h5file['x'], h5file['q'], h5file['masses'], h5file['f_tensor']

        x, q = x[()].transpose(), q[()].transpose() # np vector of x # shape now is (n_particles, dim)
        self.dim, self.n_particles = q.shape[1], q.shape[0]
        self.initialize(self.dim, self.n_particles)
        print("Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particle")
        self.x_initial.from_numpy(x)
        if flag_p2d:
            self.x.from_numpy(x+q)
        else:
            self.x.from_numpy(q)

        particle_mass = np.array(particle_mass[()].transpose().squeeze(1), dtype=np.float32) # avoid f64->f32 warning
        self.particle_mass.from_numpy(particle_mass)

        f_tensor = np.array(f_tensor[()], dtype=np.float32)
        self.F.from_numpy(f_tensor)
        self.F_disp.from_numpy(f_tensor)
        for i in range(self.n_particles):
            self.material[i] = 0
            self.v[i] = [0, 0, 0]
            if flag_p2d:
                self.F[i] += ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            else:
                self.F_disp[i] -= ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self.update_x_2d()

    def p2g2p(self, frame, dt):
        if frame == 0:
            pass # only save data, do not advance
        else:
            print("advance frame", frame)
        s1 = time.time()
        self.clear_grid()
        self.p2g(dt)
        self.grid_normalization_and_gravity(dt)
        for k in range(len(self.grid_postprocess_lead)):
            self.grid_postprocess_lead[k](self.time, dt, self.grid_v_out, self.BC_anchor_position[k], self.BC_anchor_velocity[k])#, [0.25, 0, 0], [0.01, 0, 0])
            self.BC_anchor_position[k] += dt * self.BC_anchor_velocity[k]#= [p + dt*v for p,v in zip(self.BC_anchor_position[k] ,self.BC_anchor_velocity[k])]
            # print("new anchor position is :", self.BC_anchor_position[k])
            # input()
        for collide in self.grid_postprocess:
            collide(self.time, dt, self.grid_v_out)
        self.g2p(dt)
        sN = time.time()
        print("Time for one p2g2p: ", sN - s1)
        self.time = frame * dt
        self.update_x_2d()
        diff = self.get_right_most_position()
        print("diff: ", diff - self.BC_anchor_position[0][0])

    def get_right_most_position(self):
        all_current = self.x.to_numpy()
        return float(np.amax(all_current[:,0]))


    @ti.kernel
    def clear_grid(self):
        zero = ti.Vector.zero(self.dtype, self.dim)
        for I in ti.grouped(self.grid_m):
            # If y is 3D, then I = ti.Vector([i, j, k])
            self.grid_v_in[I] = zero
            self.grid_v_out[I] = zero
            self.grid_m[I] = 0


    @ti.func
    def kirchoff_FCR(self, F, R, J, mu, la):
        #compute kirchoff stress for FCR model (remember tau = P F^T)
        # given F = U Sig V^T, R = U V^T, J = det
        return 2 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)

    @ti.kernel
    def p2g(self, dt: ti.f32):
        # f is frame number / timestep
        for p in range(0, self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.dtype) # dim(fx) = self.dim
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]

            # when to update F?
            #new_F = self.compute_von_mises(self.F_tmp[p], self.U[p], self.sig[p], self.V[p], self.yield_stress[p], self.mu[p])
            #self.F[f + 1, p] = new_F

            # for better performance, consider taking svd outside p2g loop i.e. compute svd(Fp) for all p in parallel
            U, sig, V = ti.svd(self.F[p]) # the svd of Fp at frame f
            J = 1.0
            for d in ti.static(range(self.dim)):
                J *= sig[d, d]

            # Compute Kirchoff Stress
            R = U@V.transpose()
            stress = self.kirchoff_FCR(self.F[p], R, J, self.mu[p], self.lam[p])

            # offset = [0 0 0], [0 0 1], ..., [2 2 2] in 3D
            # each dimension has stencil range 3
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                # compute weight, \Pi N_i
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                self.grid_v_in[base + offset] += weight * self.p_mass * self.v[p] # momentum transfer here
                self.grid_m[base + offset] += weight * self.p_mass # mass transfer

                # compute dweight, \Pi \nabla N_i
                dweight = ti.Vector.zero(self.dtype, self.dim)
                if self.dim==3:
                    dweight[0] = self.inv_dx * dw[offset[0]][0] *  w[offset[1]][1] *  w[offset[2]][2]
                    dweight[1] = self.inv_dx *  w[offset[0]][0] * dw[offset[1]][1] *  w[offset[2]][2]
                    dweight[2] = self.inv_dx *  w[offset[0]][0] *  w[offset[1]][1] * dw[offset[2]][2]
                else:
                    dweight[0] = self.inv_dx * dw[offset[0]][0] *  w[offset[1]][1]
                    dweight[1] = self.inv_dx *  w[offset[0]][0] * dw[offset[1]][1]

                force = - self.p_vol * stress @ dweight
                self.grid_v_in[base + offset] += dt * force # add elastic force to update velocity, don't divide by mass bc this is actually updating MOMENTUM

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: ti.f32):
        # update grid velocity at frame f
        for I in ti.grouped(self.grid_m): # loop over grid nodes
            if self.grid_m[I] > 1e-12:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = (1 / self.grid_m[I]) * self.grid_v_in[I]  # Momentum to velocity
                v_out += dt * self.gravity[None] * self.gravitational_accelaration  # gravity
                self.grid_v_out[I] = v_out


    def add_bounding_box(self):
        self.grid_postprocess.append(lambda time, dt, grid_v: self.collide_grid_bounding_box(time, dt, grid_v))

    @ti.kernel
    def collide_grid_bounding_box(self, time: ti.f32, dt: ti.f32, grid_v: ti.template()):
        for I in ti.grouped(grid_v):
            for d in ti.static(range(self.dim)):
                if I[d] < self.padding and grid_v[I][d] < 0:
                    grid_v[I][d] = 0  # Boundary conditions
                if I[d] >= self.res[d] - self.padding and grid_v[I][d] > 0:
                    grid_v[I][d] = 0

    def add_surface_collider(self,point,normal):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)
        @ti.kernel
        def collide(t: ti.f32, dt: ti.f32, grid_v: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    #if ti.static(surface == self.surface_sticky):
                    grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
        self.grid_postprocess.append(collide)

    def add_surface_leader(self, point, normal, v_BC, threshold):
        point = list(point)
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)
        v_BC = list(v_BC)
        @ti.kernel
        def lead(t: ti.f32, dt: ti.f32, grid_v: ti.template(), anchor: ti.template(), v_BC: ti.template()):# , anchor: ti.template(), v_BC: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - anchor
                n = ti.Vector(normal)
                if abs(offset.dot(n)) < threshold:
                    grid_v[I] = v_BC
        self.BC_anchor_position.append(ti.Vector(point))
        self.BC_anchor_velocity.append(ti.Vector(v_BC))
        self.grid_postprocess_lead.append(lead)

    def add_disk_leader(self, point, normal, radius, v_BC, threshold):
        point = list(point)
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)
        v_BC = list(v_BC)
        @ti.kernel
        def lead(t: ti.f32, dt: ti.f32, grid_v: ti.template(), anchor: ti.template(), v_BC: ti.template()):# , anchor: ti.template(), v_BC: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - anchor
                n = ti.Vector(normal)
                if (abs(offset.dot(n)) < threshold) and ((offset - offset.dot(n)).norm() < radius):
                    grid_v[I] = v_BC
        self.BC_anchor_position.append(ti.Vector(point))
        self.BC_anchor_velocity.append(ti.Vector(v_BC))
        self.grid_postprocess_lead.append(lead)

    @ti.kernel
    def g2p(self, dt: ti.f32):
        # advance particle at frame f
        for p in range(0, self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            dw = [fx - 1.5, -2.0 * (fx - 1), fx - 0.5]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            #new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            new_F = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
                dpos = offset.cast(self.dtype) - fx
                g_v = self.grid_v_out[base + offset]
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v

                dweight = ti.Vector.zero(self.dtype, self.dim)
                if self.dim==3:
                    dweight[0] = self.inv_dx * dw[offset[0]][0] *  w[offset[1]][1] *  w[offset[2]][2]
                    dweight[1] = self.inv_dx *  w[offset[0]][0] * dw[offset[1]][1] *  w[offset[2]][2]
                    dweight[2] = self.inv_dx *  w[offset[0]][0] *  w[offset[1]][1] * dw[offset[2]][2]
                else:
                    dweight[0] = self.inv_dx * dw[offset[0]][0] *  w[offset[1]][1]
                    dweight[1] = self.inv_dx *  w[offset[0]][0] * dw[offset[1]][1]
                new_F += g_v.outer_product(dweight)
                #new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p]= new_v

            self.x[p] = self.x[p] + dt * self.v[p]
            #self.x[p] = ti.max(ti.min(self.x[p] + dt * self.v[p], 1.-3*self.dx), 0.)
            # advection and preventing it from overflow, fundamentally is x0 + dt * v

            # update deformation gradient
            # if no plasciticy, this will be the new F
            F_trial = self.compute_F_trial(p, new_F, dt)
            if self.use_plasticity_vm:
                self.von_mises_return_mapping(p, F_trial)
            else:
                self.F[p] = F_trial

            self.F_disp[p] = self.F[p] -  ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.u[p] = self.x[p] - self.x_initial[p]


    @ti.func
    def compute_F_trial(self, p, new_F, dt):
        # if no plasticity, then this is F.
        # otherwise this is F_trial
        F_trial = (ti.Matrix.identity(self.dtype, self.dim) + (dt * new_F)) @ self.F[p] #updateF (explicitMPM way)
        return F_trial

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)


    @ti.func
    def von_mises_return_mapping(self, p, F_trial):
        # F_trial to F_elastic
        U_trial, sig_trial, V_trial = ti.svd(F_trial) # the svd of Fp at frame f
        # follow PlasticineLab, which follows Ming Gao 2017, which follows Klar 2016
        epsilon = ti.Vector.zero(self.dtype, self.dim)
        sig_trial = ti.max(sig_trial, 0.05) # add this to prevent NaN in extrem cases
        if ti.static(self.dim == 2):
            epsilon = ti.Vector([ti.log(sig_trial[0, 0]), ti.log(sig_trial[1, 1])])
        else:
            epsilon = ti.Vector([ti.log(sig_trial[0, 0]), ti.log(sig_trial[1, 1]), ti.log(sig_trial[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / self.dim)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - self.yield_stress[p] / (2 * self.mu[p]) # yield stress is parameter threshold
        if delta_gamma > 0:  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig_elastic = self.make_matrix_from_diag(ti.exp(epsilon)) # projected eigenvalues
            F_trial = U_trial @ sig_elastic @ V_trial.transpose()

        self.F[p] = F_trial

    def particle_position2obj(self, fullfilename):
        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        objfile = open(fullfilename, 'w')
        for i in range(self.n_particles):
            line =  "v " + str(self.x[i][0]) + " " + str(self.x[i][1]) + " " + str(self.x[i][2])
            objfile.write(line)
            objfile.write('\n')

        print('taichi mpm solver writes current position at ', fullfilename)

    def save_data_at_frame(self, dir_name, frame, flag_p2d = True, save_to_h5 = True, save_to_obj = True):
        os.umask(0)
        os.makedirs(dir_name, 0o777, exist_ok=True)
        fullfilename = dir_name + '/h5_f_' + str(frame).zfill(10) + '.h5'
        # print("fullfilename is ", fullfilename)


        #----------- Initial position --------------
        x_initial_np = self.x_initial.to_numpy()
        x_initial_np = x_initial_np.transpose()
        #----------- Deformed position / Displacement --------------
        x_np = self.x.to_numpy() # x_np has dimension (dim, n_particles)
        x_np = x_np.transpose()
        #----------- velocity --------------
        v_np = self.v.to_numpy() # x_np has dimension (dim, n_particles)
        v_np = v_np.transpose()
        ########## Insert saving to obj here since obj always requires a deformed position
        if save_to_obj:
            fullfilename_obj = fullfilename[:-2]+'obj'
            if os.path.exists(fullfilename_obj):
                os.remove(fullfilename_obj)
            objfile = open(fullfilename_obj, 'w')
            for k in range(x_np.shape[1]): # loop over all particles
                line = "v " + str(x_np[0,k]) + " " + str(x_np[1,k]) + " " + str(x_np[2,k])
                objfile.write(line)
                objfile.write('\n')
            print("save siumlation data at frame ", frame, " to ", fullfilename_obj)
        ########## Insert saving to obj here since obj always requires a deformed position
        if flag_p2d:
            x_np = x_np - x_initial_np
        #----------- Time --------------
        currentTime = np.array([self.time])
        currentTime = currentTime.reshape(1,1) # need a 1by1 matrix
        #----------- Particle mass --------------
        p_mass_np = self.particle_mass.to_numpy()
        p_mass_np = p_mass_np.reshape(1, self.n_particles)
        #----------- Deformation gradient, dxdX / dudX --------------
        if flag_p2d:
            f_tensor_np = self.F_disp.to_numpy() # dimension (n_particles, 3, 3)
        else:
            f_tensor_np = self.F.to_numpy() # dimension (n_particles, 3, 3)
        f_tensor_np = f_tensor_np.reshape(-1,9) # (n,3,3) -> (n,9), row_major
        f_tensor_np = f_tensor_np.transpose() # (9,n)
        if save_to_h5:
            if os.path.exists(fullfilename):
                os.remove(fullfilename)
            newFile = h5py.File(fullfilename, "w")
            newFile.create_dataset("x", data=x_initial_np) # initial position
            newFile.create_dataset("q", data=x_np) # deformed position / displacement
            newFile.create_dataset("time", data=currentTime) # current time
            newFile.create_dataset("masses", data=p_mass_np) # particle mass
            newFile.create_dataset("f_tensor", data=f_tensor_np) # deformation grad
            newFile.create_dataset("v", data=v_np) # particle velocity
            print("save siumlation data at frame ", frame, " to ", fullfilename)

    # # override this one if possible, and try always use this one
    # def advance_at_frame(self, frame, save_to_h5 = False, save_to_obj = False, dirname = None):
    #
    #     if frame == 0:
    #         if save_to_h5:
    #             self.save_data_at_frame(dirname, 0, True, save_to_obj)
    #     else:
    #         self.p2g2p(frame) # override this one if needed
    #         if save_to_h5:
    #             self.save_data_at_frame(dirname, frame, True, save_to_obj)
