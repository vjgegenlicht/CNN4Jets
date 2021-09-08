import numpy as np
import matplotlib.pyplot as plt
import os

def jet_preprocessing(eta, phi, intensity, params):

    eta_preproc = eta.copy()
    phi_preproc = phi.copy() 

    jet_preproc_steps = params.get('jet_preproc_steps', [])

    # Align phi along leading jet
    if 'phi_alignment' in jet_preproc_steps:
        phi_preproc                       -= phi_preproc[:,0,None]
        phi_preproc[phi_preproc < -np.pi] += 2*np.pi
        phi_preproc[phi_preproc > np.pi]  -= 2*np.pi
    
    # Centering along each axis
    if 'centering' in jet_preproc_steps:
        eta_center  = np.average( eta_preproc, axis=1, weights=intensity )
        phi_center  = np.average( phi_preproc, axis=1, weights=intensity )

        eta_preproc -= eta_center[:,None]
        phi_preproc -= phi_center[:,None]
        
    return eta_preproc, phi_preproc

def image_preprocessing(images, params):

    image_preproc_steps = params.get('image_preproc_steps', [])

    # Normalizing the Images
    if 'normalization' in image_preproc_steps:
        norm    = np.clip( np.sum( images, axis=(1,2) ), 1.e-5, np.inf )
        images  = images/norm[:,None,None]
    
    return images


def make_image(eta, phi, intensity, params):
    '''Bins the (eta, phi)-plane and computes the integrated bin-wise intensity.
        args: 
            eta         : [array]   array of eta values
            phi         : [array]   array of phi values
            intensity   : [array]   array of intensities (pt, E or whatever)
            params      : [dict]    dictionary holding the parameters
    '''
    # Parameters
    n_pixel     = params.get('n_pixel', 10)
    eta_range   = params.get('eta_range', [-0.58,0.58])
    phi_range   = params.get('phi_range', [-0.7,0.7]) 
    eta_pixels  = np.linspace(eta_range[0], eta_range[1], n_pixel)
    phi_pixels  = np.linspace(phi_range[0], phi_range[1], n_pixel)

    # Image
    n_events    = eta.shape[0]
    n_const     = eta.shape[1]
    calo        = np.zeros((n_events, len(eta_pixels), len(phi_pixels)))
    
    for i in range(n_const):

        eta_i       = eta[:, i]
        phi_i       = phi[:, i]
        intensity_i = intensity[:, i]

        in_grid = ~((eta_i < eta_pixels[0]) | (eta_i > eta_pixels[-1]) | (phi_i < phi_pixels[0]) | (phi_i > phi_pixels[-1]))
        xcoords = np.argmax(eta_i[:,None] < eta_pixels[None,:],axis=1)
        ycoords = np.argmax(phi_i[:,None] < phi_pixels[None,:],axis=1)
        ncoords = np.arange(n_events)

        calo[ncoords[in_grid], ycoords[in_grid], xcoords[in_grid]] += intensity_i[in_grid]

    return calo

def compute_m(E, px, py, pz):
    '''Computes the invariant mass.
        args:
            E       : [array]   array of energies
            px      : [array]   array of momenta in x-direction
            py      : [array]   array of momenta in y-direction
            pz      : [array]   array of momenta in z-direction
    '''
    return np.sqrt(E**2-px**2-py**2-pz**2)
    
def compute_pt(px, py):
    ''' Computes transverse momentum from momenta in x and y direction.
        args:
            px      : [array]   array of momenta in x-direction
            py      : [array]   array of momenta in y-direction
    '''
    return np.sqrt( px**2+py**2 )

def compute_eta(pt, pz):
    '''Computes eta from transverse momentum and momentum in z-direction.
        args:
            pt      : [array]   array of transverse momenta
            pz      : [array]   array of momenta in z-direction
    '''
    small       = 1e-10
    small_pt    = (np.abs(pt) < small)
    small_pz    = (np.abs(pz) < small)
    not_small   = ~(small_pt | small_pz)

    theta               = np.arctan(pt[not_small]/pz[not_small])
    theta[theta < 0]    += np.pi

    eta             = np.zeros_like(pt)
    eta[small_pz]   = 0
    eta[small_pt]   = 1e-10
    eta[not_small]  = np.log(np.tan(theta/2))
    return eta

def compute_phi(px, py):
    ''' Computes phi from momenta in x and y direction.
        args:
            px      : [array]   array of momenta in x-direction
            py      : [array]   array of momenta in y-direction
    '''
    phi                 = np.arctan2(py,px)
    phi[phi < 0]        += 2*np.pi
    phi[phi > 2*np.pi]  -= 2*np.pi
    phi                 = phi - np.pi
    return phi

def plot_images(images, set, params):
    '''Plots some of the generated detector images.
        args:
            images  : [array]   array holding the images
            params  : [dict]    dictionary holding the parameters
    '''
    image_output_path = params.get('image_output_path', './images/')
    os.makedirs(image_output_path + '/images/' + set, exist_ok=True) 

    n_pixel     = params.get('n_pixel', 10)
    eta_range   = params['eta_range']
    phi_range   = params['phi_range']

    plt.figure(figsize=(6, 6), dpi=160)    
    #plt.rc("text", usetex=True)
    #plt.rc("font", family="serif")
    #plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
    plt.imshow(np.mean(images, axis=0))
    plt.title("Average", fontsize=16)
    plt.xlabel("eta", fontsize=14)
    plt.ylabel("phi", fontsize=14)
    plt.xticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(eta_range[0],eta_range[1], 5),1))
    plt.yticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(phi_range[0],phi_range[1], 5),1))
    plt.savefig(image_output_path + "/" + set + "/average", dpi=160, format="png")

    for i in range(params.get('n_images', 10)):
        plt.imshow(images[i])
        plt.xlabel("eta", fontsize=14)
        plt.ylabel("phi", fontsize=14)
        plt.xticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(eta_range[0],eta_range[1], 5),1))
        plt.yticks(np.linspace(0,n_pixel-1,5), np.round(np.linspace(phi_range[0],phi_range[1], 5),1))
        plt.savefig(image_output_path + "/" + set + "/img_{}.png".format(i), dpi=160, format="png")