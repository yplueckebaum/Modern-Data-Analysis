from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sksfa
import pickle


def load_data():
    with open('excersise2.pkl', 'rb') as f:
        savedict = pickle.load(f)
    mean = savedict['mean']
    components = savedict['components']
    video_pca = savedict['video_pca']
    return mean, components, video_pca


def display_video(video, reprojection_function=None, original=None, original_reprojection_function=None):
    assert original is None or original_reprojection_function is not None
    # Display video using cv2
    import cv2
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    while True:
        if original is not None:
            cv2.namedWindow('Video Original', cv2.WINDOW_NORMAL)
            for frame, frame_o in zip(video, original):
                if reprojection_function is not None:
                    frame = reprojection_function(frame)
                    frame[frame < 0] = 0
                    frame[frame > 255] = 255
                if original_reprojection_function is not None:
                    frame_o = original_reprojection_function(frame_o)
                    frame_o[frame_o < 0] = 0
                    frame_o[frame_o > 255] = 255
                cv2.imshow('Video', frame.reshape((128, 128)).astype('uint8'))
                cv2.imshow('Video Original', frame_o.reshape((128, 128)).astype('uint8'))
                key = cv2.waitKey(50)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
        else:
            for frame in video:
                if reprojection_function is not None:
                    frame = reprojection_function(frame)
                    frame[frame < 0] = 0
                    frame[frame > 255] = 255
                cv2.imshow('Video', frame.reshape((128, 128)).astype('uint8'))
                key = cv2.waitKey(50)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return


if __name__ == '__main__':
    mean, components, x = load_data()
    # write your code here
    # a) PCA reprojection
    print('Mean: ', mean)
    print('Components: ', components)
    print('Components shape: ', components.shape)
    print('Variance: ', np.var(x, axis=0))
    print('Mean: ', np.mean(x, axis=0))
    print('Shape: ', x.shape)

    # Plot
    # Transform one frame to the original space
    frame = x[0]
    frame = np.dot(frame, components) + mean
    frame = frame.reshape((128, 128))
    plt.imshow(frame, cmap='gray')
    plt.show()

    pca_reconstruction = lambda f: np.dot(f, components) + mean
    # display_video(x, pca_reconstruction)

    # What object is shown in the video/images?
    # A rotating shovel is shown.

    # b) Applying SFA
    # Apply SFA to the data and display the result.
    sfa = sksfa.SFA(n_components=2)
    sfa_x = sfa.fit_transform(x)
    print('SFA shape: ', sfa_x.shape)
    print('SFA variance: ', np.var(sfa_x, axis=0))
    print('SFA mean: ', np.mean(sfa_x, axis=0))

    # Plot
    plt.plot(sfa_x[:, 0])
    plt.plot(sfa_x[:, 1])
    plt.show()

    # Reconstruct
    w, b = sfa.affine_parameters()
    sfa_reconstruction = lambda x: pca_reconstruction(np.dot(np.linalg.pinv(w), x - b))

    # display_video(sfa_x, sfa_reconstruction)
    display_video(sfa_x, sfa_reconstruction, x, pca_reconstruction)

    # What do these features characterize?
    # The first features represents the rotating main part of the shovel (?),
    # the second features represents the appearance and disappearance of the flat inner
    # part of the shovel, which is visible when the shovel is rotated half way and is visibly darker.
    # (See video, compare to the plot of the second features and video with only one features.)

    # What do you see if you plot these features against each other?
    # The first feature is steadily decreasing, the second feature has its peak at the middle of the video,
    # decreasing before and after that, looking like a single phase of a -cosine curve.
