import numpy as np
from sklearn.neighbors import NearestNeighbors

class Registration:
    """
    A comprehensive point cloud registration class for 3D computer vision applications.
    
    This class provides methods for both coarse and fine registration of multiple point clouds:
    - Coarse registration: Circular arrangement with rotation-based alignment
    - Fine registration: ICP (Iterative Closest Point) algorithm with sequential and pairwise options
    """
    
    def __init__(self):
        """Initialize the Registration class."""
        pass
    
    # ============================================================================
    # COARSE REGISTRATION FUNCTIONS
    # ============================================================================
    
    def calculate_centroid(self, PC):
        centroid = np.mean(PC, axis=0)  # [cx, cy, cz]
        return centroid

    def center_PC(self, PC, centroid):
        centered_pc = PC - centroid
        return centered_pc

    def get_rotation_matrix_y(self, angle_rad):
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        R = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]])
            
        return R    

    def arrange_views_in_circle(self, point_clouds, angles, radius):
        # 1. Calculate the central point (the "red dot") for the entire construction.
        all_points_combined = np.vstack(point_clouds)
        construction_center = self.calculate_centroid(all_points_combined)
        print(f"Calculated construction center (Red Dot): {construction_center}")

        arranged_pcs = []
        # 2. Loop through each view to orient it rotation only
        for pc, angle in zip(point_clouds, angles):
            print(f"\nArranging {np.degrees(angle):.0f}° view...")

            # 3. First, orient the point cloud correctly.
            #    - Center it at the origin to rotate it around its own center.
            #    - Apply its forward rotation to align it with the world frame.
            current_centroid = self.calculate_centroid(pc)
            centered_pc = self.center_PC(pc, current_centroid)
            
            # Apply rotationto correct camera rotation
            R = self.get_rotation_matrix_y(angle)
            rotated_pc = np.dot(centered_pc, R.T)

            # 4. Store rotated pc to transformed
            transformed_pc = rotated_pc
            arranged_pcs.append(transformed_pc)
            
        return arranged_pcs, construction_center

    def check_alignment_quality(self, transformed_pcs):
        centroids = [self.calculate_centroid(pc) for pc in transformed_pcs]
        
        # calculate centroid distances
        centroid_distances = []
        reference_centroid = centroids[0]
        
        for centroid in centroids[1:]:
            distance = np.linalg.norm(centroid - reference_centroid)
            centroid_distances.append(distance)
            
        return centroid_distances
    
    # ============================================================================
    # FINE REGISTRATION FUNCTIONS
    # ============================================================================
    
    def find_correspondences(self, source_pc, target_pc, max_distance=0.01):
        """
        Step 1: Function to find the Correspondences
        """
        # Build kd_tree for nearest neighbor search
        kd_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_pc)
        
        # Find nearest neighbors for each source point
        distances, indices = kd_tree.kneighbors(source_pc)
        
        # Filter correspondences by max dist
        valid_mask = distances.flatten() < max_distance
        
        valid_source_points = source_pc[valid_mask]
        valid_target_indices = indices[valid_mask].flatten()
        valid_target_points = target_pc[valid_target_indices]
        valid_distances = distances[valid_mask].flatten()
        
        correspondences = list(zip(
            np.where(valid_mask)[0],
            valid_target_indices,
            valid_distances))
            
        return correspondences, valid_source_points, valid_target_points
        
    def estimate_transformation(self, source_points, target_points):
        """
        Step 2: Estimate optimal rotation and translation using SVD-based method
        """
        # calc centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        # center the clouds
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # compute cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # compute rotation matrix
        R = Vt.T @ U.T
        
        # det(R) = 1
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # compute translation
        trans = target_centroid - R @ source_centroid
        
        # calc RMSE for convergence checking
        transformed_source = (R @ source_points.T).T + trans
        squared_errors = np.sum((transformed_source - target_points)**2, axis=1)
        rmse = np.sqrt(np.mean(squared_errors))
        
        return R, trans, rmse
        
    def transform_PC(self, PC, R, t):
        """
        Step 3: Applying rotation and translation to PC
        """
        transformed_pc = (R @ PC.T).T + t  # new_point = R * old_point + t
        return transformed_pc
        
    def icp_registration(self, source_pc, target_pc, max_iterations=200, tolerance=1e-6, max_corr_dist=0.01):
        """
        Step 4: ICP registration algorithm
        """
        # initialization
        current_source = source_pc.copy()
        cumulative_R = np.eye(3)
        cumulative_t = np.zeros(3)
        convergence_history = []
        prev_rmse = float('inf')
        
        rmse = 0.0
        rmse_change = float('inf')
        converged = False
        iterations_completed = 0
        
        # Loop that conducts the full implementation
        for i in range(max_iterations):
            iterations_completed = i + 1
            
            # step 1: find corr
            corr, valid_source, valid_target = self.find_correspondences(current_source, target_pc, max_corr_dist)
            # Check corr validity
            if len(corr) < 10:  # Min threshold
                print(f"Warning: Only {len(corr)} correspondences found. Stopping.")
                break
            
            # step 2: estimate transformation
            R, t, rmse = self.estimate_transformation(valid_source, valid_target)
            
            # step 3: apply transformation
            current_source = self.transform_PC(current_source, R, t)
            
            # update cumulative transformation
            cumulative_R = R @ cumulative_R
            cumulative_t = R @ cumulative_t + t
            
            # update convergence history
            convergence_history.append(rmse)
            
            # check convergence
            rmse_change = abs(prev_rmse - rmse)
            if rmse_change < tolerance:
                print(f"\nConverged after {i + 1} iterations (RMSE change: {rmse_change:.2e})")
                converged = True
                break
                
            prev_rmse = rmse
            
        # Prepare final results
        final_transformation = {
            'R': cumulative_R,
            't': cumulative_t,
            'rmse': rmse,
            'iterations': iterations_completed,
            'converged': converged
        }
        print(f"ICP Registration Complete:")
        print(f"  Final RMSE: {rmse*1000:.3f}mm")
        print(f"  Iterations: {iterations_completed}")
        print(f"  Converged: {final_transformation['converged']}")
        
        return final_transformation, current_source, convergence_history
            
    def sequential_icp_registration(self, transformed_pcs, icp_params=None):
        """
        Step 5: Apply ICP registration to multiple PC's (Sequential approach)
        """
        # Initialize icp Parameters if non provided
        if icp_params is None:
            icp_params = {
                'max_iterations': 200,
                'tolerance': 1e-6,
                'max_corr_dist': 0.02
            }
            
        # Init ref PC
        fine_registered_pcs = [transformed_pcs[0].copy()]
        transformations = [{'R': np.eye(3), 't': np.zeros(3)}]
        registration_stats = []
        
        labels = ['0°', '90°', '180°', '270°'] 
        
        # Sequential loop
        for i in range(1, len(transformed_pcs)):
            # use accumulated registered points as target_source
            accumulated_target = np.vstack(fine_registered_pcs)
            current_source = transformed_pcs[i]
            
            # Calculate initial alignment error
            source_centroid = self.calculate_centroid(current_source)
            target_centroid = self.calculate_centroid(accumulated_target)
            initial_error = np.linalg.norm(source_centroid - target_centroid)
            print(f"Initial centroid distance: {initial_error*1000:.3f}mm")
            
            # apply icp
            transformation, registered_source, convergence = self.icp_registration(
                current_source, accumulated_target, 
                icp_params['max_iterations'], 
                icp_params['tolerance'], 
                icp_params['max_corr_dist'])
            
            # store results
            fine_registered_pcs.append(registered_source)
            transformations.append(transformation)
            registration_stats.append({
                'view': labels[i],
                'final_rmse_mm': transformation['rmse'] * 1000,
                'iterations': transformation['iterations'],
                'converged': transformation['converged'],
                'convergence_history': convergence
            })
            
        # summary stats
        total_points = sum(len(pc) for pc in fine_registered_pcs)
        avg_rmse = np.mean([stat['final_rmse_mm'] for stat in registration_stats])

        print(f"\nTotal registered points: {total_points:,}")
        print(f"Average final RMSE: {avg_rmse:.3f}mm")
        print(f"All views converged: {all(stat['converged'] for stat in registration_stats)}")
        
        return fine_registered_pcs, transformations, registration_stats

    def pairwise_icp_registration(self, transformed_pcs, icp_params=None):
        """
        New function approach using pairwise registration based on the 0° reference PC
        Register each view directly to the reference (0°) view instead of sequential registration.
        This avoids error accumulation and handles distant point clouds better.
        
        """
        if icp_params is None:
            icp_params = {
                'max_iterations': 200,
                'tolerance': 1e-6,
                'max_corr_dist': 0.05  # 50mm - adjust as needed
            }
            
        fine_registered_pcs = [transformed_pcs[0].copy()]  # Reference view unchanged
        transformations = [{'R': np.eye(3), 't': np.zeros(3)}]
        registration_stats = []
        
        labels = ['0°', '90°', '180°', '270°']
        reference_pc = transformed_pcs[0]  # Always use 0° as reference
        
        print(f"Pairwise ICP Parameters:")
        print(f"  Max iterations: {icp_params['max_iterations']}")
        print(f"  Tolerance: {icp_params['tolerance']}")
        print(f"  Max correspondence distance: {icp_params['max_corr_dist']*1000:.1f}mm")
        print(f"  Reference view: {labels[0]} ({len(reference_pc):,} points)")
        print()
        
        # Register each view to the reference
        for i in range(1, len(transformed_pcs)):
            print(f"{'='*50}")
            print(f"REGISTERING {labels[i]} TO REFERENCE ({labels[0]})")
            print(f"{'='*50}")
            
            current_source = transformed_pcs[i]
            
            print(f"Source points ({labels[i]}): {len(current_source):,}")
            print(f"Target points ({labels[0]}): {len(reference_pc):,}")
            
            # Calculate initial alignment error
            source_centroid = self.calculate_centroid(current_source)
            target_centroid = self.calculate_centroid(reference_pc)
            initial_error = np.linalg.norm(source_centroid - target_centroid)
            print(f"Initial centroid distance: {initial_error*1000:.3f}mm")
            
            # Apply ICP registration
            transformation, registered_source, convergence = self.icp_registration(
                current_source, reference_pc,  # Always register to reference
                icp_params['max_iterations'], 
                icp_params['tolerance'], 
                icp_params['max_corr_dist']
            )
            
            # Store results
            fine_registered_pcs.append(registered_source)
            transformations.append(transformation)
            registration_stats.append({
                'view': labels[i],
                'initial_error_mm': initial_error * 1000,
                'final_rmse_mm': transformation['rmse'] * 1000,
                'iterations': transformation['iterations'],
                'converged': transformation['converged'],
                'convergence_history': convergence
            })
            
            print(f"Registration complete for {labels[i]} view")
            print()
            
        # Summary statistics
        total_points = sum(len(pc) for pc in fine_registered_pcs)
        avg_rmse = np.mean([stat['final_rmse_mm'] for stat in registration_stats])
        all_converged = all(stat['converged'] for stat in registration_stats)

        print(f"{'='*50}")
        print(f"PAIRWISE ICP REGISTRATION COMPLETE")
        print(f"{'='*50}")
        print(f"Total registered points: {total_points:,}")
        print(f"Average final RMSE: {avg_rmse:.3f}mm")
        print(f"All views converged: {all_converged}")
        print()
        
        # Print detailed results
        print("Detailed Results:")
        for stat in registration_stats:
            print(f"  {stat['view']} -> {labels[0]}:")
            print(f"    Initial error: {stat['initial_error_mm']:.3f}mm")
            print(f"    Final RMSE: {stat['final_rmse_mm']:.3f}mm")
            print(f"    Iterations: {stat['iterations']}")
            print(f"    Converged: {stat['converged']}")
            print(f"    Improvement: {stat['initial_error_mm'] - stat['final_rmse_mm']:.3f}mm")
            print()
        
        return fine_registered_pcs, transformations, registration_stats
