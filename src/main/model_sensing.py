import os
import sys
import time
import gc
import matplotlib.pyplot as plt
# to make run from console for module import
sys.path.append(os.path.abspath(".."))

# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.random.set_seed(2023)
try:
    from IPython import get_ipython

    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except:  # noqa E722
    from tqdm import tqdm

from main.config_sensing import Config
from main.dataset import Dataset
from main.discriminator import Discriminator
from main.generator_sensing_ori import Generator
from main.model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues

import tensorflow.compat.v1.losses as v1_loss


class ExceptionHandlingIterator:
    """This class was introduced to avoid tensorflow.python.framework.errors_impl.InvalidArgumentError
        thrown while iterating over the zipped datasets.

        One assumption is that the tf records contain one wrongly generated set due to following error message:
            Expected begin[1] in [0, 462], but got -11 [[{{node Slice}}]] [Op:IteratorGetNextSync]
    """

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iter.__next__()
        except StopIteration as e:
            raise e
        except Exception as e:
            print(e)
            return self.__next__()


class Model:

    def __init__(self, display_config=True):
        self.config = Config()
        self.config.save_config()
        if display_config:
            self.config.display()

        self._build_model()
        self._setup_summary()

    def _build_model(self):
        print('building model...\n')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        gen_input = ((self.config.BATCH_SIZE,) + self.config.ENCODER_INPUT_SHAPE)

        self.generator = Generator()
        self.generator.build(input_shape=gen_input)
        # self.generator.summary()
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=self.config.GENERATOR_LEARNING_RATE)

        if not self.config.ENCODER_ONLY:
            disc_input = (self.config.BATCH_SIZE, self.config.NUM_JOINTS * 9 + self.config.NUM_SHAPE_PARAMS)

            self.discriminator = Discriminator()
            self.discriminator.build(input_shape=disc_input)
            self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=self.config.DISCRIMINATOR_LEARNING_RATE)

        # setup checkpoint
        self.checkpoint_prefix = os.path.join(self.config.LOG_DIR, "ckpt")
        if not self.config.ENCODER_ONLY:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             discriminator=self.discriminator,
                                             generator_opt=self.generator_opt,
                                             discriminator_opt=self.discriminator_opt)
        else:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             generator_opt=self.generator_opt)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.config.LOG_DIR, max_to_keep=2)

        # if a checkpoint exists, restore the latest checkpoint.
        self.restore_check = None
        if self.checkpoint_manager.latest_checkpoint:
            restore_path = self.config.RESTORE_PATH
            if restore_path is None:
                restore_path = self.checkpoint_manager.latest_checkpoint

            self.restore_check = checkpoint.restore(restore_path).expect_partial()
            print('Checkpoint restored from {}'.format(restore_path))

    def _setup_summary(self):
        self.summary_path = os.path.join(self.config.LOG_DIR, 'hmr2.0', '3D_{}'.format(self.config.USE_3D))
        self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        self.generator_loss_log = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.psnr_log = tf.keras.metrics.Mean('psnr', dtype=tf.float32)

        self.generator_loss_test_log = tf.keras.metrics.Mean('generator_loss_test', dtype=tf.float32)
        self.psnr_test_log = tf.keras.metrics.Mean('psnr_test', dtype=tf.float32)


    ############################################################
    #  Train/Val
    ############################################################

    def train(self):
        # Place tensors on the CPU
        # with tf.device('/CPU:0'):
        dataset = Dataset()

        with tf.device('/CPU:0'):
            ds_train = dataset.get_train()
            ds_smpl = dataset.get_smpl()
            ds_val = dataset.get_val()

        start = 1
        if self.config.RESTORE_EPOCH:
            start = self.config.RESTORE_EPOCH

        for epoch in range(start, self.config.EPOCHS + 1):
            gc.collect()
            start = time.time()
            print('Start of Epoch {}'.format(epoch))

            dataset_train = ExceptionHandlingIterator(tf.data.Dataset.zip((ds_train, ds_smpl)))
            total = int(self.config.NUM_TRAINING_SAMPLES / self.config.BATCH_SIZE)

            for image_data, theta in tqdm(dataset_train, total=total, position=0, desc='training'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._train_step(images, kp2d, kp3d, has3d, theta)


            self._log_train(epoch=epoch)

            total = int(self.config.NUM_VALIDATION_SAMPLES / self.config.BATCH_SIZE)
            for image_data in tqdm(ds_val, total=total, position=0, desc='validate'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._val_step(images, kp2d, kp3d, has3d)


            self._log_val(epoch=epoch)

            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

            # saving (checkpoint) the model every 5 epochs
            if epoch % 1 == 0:
                print('saving checkpoint\n')
                self.checkpoint_manager.save(epoch)

        self.summary_writer.flush()
        self.checkpoint_manager.save(self.config.EPOCHS + 1)

    @tf.function
    def _train_step(self, images, kp2d, kp3d, has3d, theta):
        tf.keras.backend.set_learning_phase(1)
        batch_size = images.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generator_outputs, rate = self.generator(images, training=True)
            generator_outputs = self.generator(images, training=True)
            # only use last computed theta (from iterative feedback loop)
            sq_err = tf.math.squared_difference(images, generator_outputs)
            distortion_loss = tf.reduce_mean(sq_err)
            psnr = tf.reduce_mean(tf.image.psnr(images, generator_outputs, max_val=1.0))
            # use all poses and shapes from iterative feedback loop


            generator_loss = distortion_loss


        generator_grads = gen_tape.gradient(generator_loss, self.generator.trainable_variables)

        self.generator_opt.apply_gradients(zip(generator_grads, self.generator.trainable_variables))

        self.generator_loss_log.update_state(generator_loss)
        self.psnr_log.update_state(psnr)


    def accumulate_fake_disc_input(self, generator_outputs):
        fake_poses, fake_shapes = [], []
        for output in generator_outputs:
            fake_poses.append(output[3])
            fake_shapes.append(output[4])
        # ignore global rotation
        fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self.config.NUM_JOINTS_GLOBAL, 9])[:, 1:, :]
        fake_poses = tf.reshape(fake_poses, [-1, self.config.NUM_JOINTS * 9])
        fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self.config.NUM_SHAPE_PARAMS])

        fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
        return fake_disc_input

    def accumulate_real_disc_input(self, theta):
        real_poses = theta[:, :self.config.NUM_POSE_PARAMS]
        # compute rotations matrices for [batch x K x 9] - ignore global rotation
        real_poses = batch_rodrigues(real_poses)[:, 1:, :]
        real_poses = tf.reshape(real_poses, [-1, self.config.NUM_JOINTS * 9])
        real_shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]

        real_disc_input = tf.concat([real_poses, real_shapes], 1)
        return real_disc_input

    def _log_train(self, epoch):
        template = 'Generator Loss: {} PSNR: {}'
        print(template.format(self.generator_loss_log.result(), self.psnr_log.result()))

        with self.summary_writer.as_default():
            tf.summary.scalar('generator_loss', self.generator_loss_log.result(), step=epoch)
            tf.summary.scalar('PSNR', self.generator_loss_log.result(), step=epoch)

        self.generator_loss_log.reset_states()
        self.psnr_log.reset_states()


    @tf.function
    def _val_step(self, images, kp2d, kp3d, has3d):
        tf.keras.backend.set_learning_phase(0)


        results = self.generator(images, training=False)
        # only use last computed theta (from accumulated iterative feedback loop)
        sq_err = tf.math.squared_difference(images, results)
        distortion_loss = tf.reduce_mean(sq_err)
        psnr = tf.reduce_mean(tf.image.psnr(images, results, max_val=1.0))
        # use all poses and shapes from iterative feedback loop

        generator_loss = distortion_loss


        self.generator_loss_test_log.update_state(generator_loss)
        self.psnr_test_log.update_state(psnr)



    def _log_val(self, epoch):
        print('MSE: {}'.format(self.generator_loss_test_log.result()))
        print('PSNR: {}'.format(self.psnr_test_log.result()))


        with self.summary_writer.as_default():
            tf.summary.scalar('MSE', self.generator_loss_test_log.result(), step=epoch)
            tf.summary.scalar('PSNR', self.psnr_test_log.result(), step=epoch)

        self.generator_loss_test_log.reset_states()
        self.psnr_test_log.reset_states()

    ############################################################
    #  Test
    ############################################################

    def test(self, return_kps=True):
        """Run evaluation of the model
        Specify LOG_DIR to point to the saved checkpoint directory

        Args:
            return_kps: set to return keypoints - default = False
        """

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        # Place tensors on the CPU
        with tf.device('/CPU:0'):
            dataset = Dataset()
            ds_test = dataset.get_test()

        start = time.time()
        print('Start of Testing')
        # prior_learnt=self.generator.compression_trainer.prior
        # _ = tf.linspace(-6., 6., 501)[:, None]
        # plt.plot(_, prior_learnt.prob(_))
        # plt.savefig('learnt_prior_latent10.png')
        # plt.show()

        # mpjpe, mpjpe_aligned, kps2d_pred, kps2d_real = [], [], [], []
        psnr =[]

        total = int(self.config.NUM_TEST_SAMPLES / self.config.BATCH_SIZE)
        for image_data in tqdm(ds_test, total=total, position=0, desc='testing'):
            image, kp2d, kp3d, has_3d = image_data[0], image_data[1], image_data[2], image_data[3]
            # kp2d_mpjpe, kp2d_mpjpe_aligned, predict_kp2d, real_kp2d = self._test_step(image, kp2d[:,:,:2], return_kps=return_kps)
            psnr_ = self._test_step(image, kp2d, kp3d, has_3d, return_kps=return_kps)


            psnr.append(psnr_)

        print('Time taken for testing {} sec\n'.format(time.time() - start))

        def convert(tensor, num=None, is_kp=False):
            if num is None:
                num = self.config.NUM_KP2D
            if is_kp:
                return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num, 3]))

            return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num]))

        # mpjpe, mpjpe_aligned= convert(mpjpe), convert(mpjpe_aligned)
        # result_dict = {"kp2d_mpjpe": convert(mpjpe), "kp2d_mpjpe_aligned": mpjpe_aligned}
        result_dict = {"psnr": tf.reduce_mean(tf.stack(psnr))}



        return result_dict

    # @tf.function
    def _test_step(self, image, kp2d, kp3d, has3d, return_kps=False):
        tf.keras.backend.set_learning_phase(0)

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)
            kp2d = tf.expand_dims(kp2d, 0)

        generator_outputs = self.generator(image, training=False)
        # only use last computed theta (from iterative feedback loop)
        sq_err = tf.math.squared_difference(image, generator_outputs)
        distortion_loss = tf.reduce_mean(sq_err)
        psnr = tf.reduce_mean(tf.image.psnr(image, generator_outputs, max_val=1.0))
        # use all poses and shapes from iterative feedback loop



        return psnr

        # factor = tf.constant(1000, tf.float32)
        # kp2d, kp2d_predict = kp2d * factor, kp2d_pred * factor  # convert back from m -> mm
        # kp2d_predict = kp2d_predict[:, :self.config.NUM_KP2D, :]
        #
        # real_kp2d = batch_align_by_pelvis(kp2d)
        # predict_kp2d = batch_align_by_pelvis(kp2d_predict)
        #
        # kp2d_mpjpe = tf.norm(real_kp2d - predict_kp2d, axis=2)
        #
        # aligned_kp2d = batch_compute_similarity_transform(real_kp2d, predict_kp2d)
        # kp2d_mpjpe_aligned = tf.norm(real_kp2d - aligned_kp2d, axis=2)

        # if return_kps:
        #     return kp2d_mpjpe, kp2d_mpjpe_aligned, predict_kp2d, real_kp2d
        #
        # return kp2d_mpjpe, kp2d_mpjpe_aligned, None, None

    ############################################################
    #  Detect/Single Inference
    ############################################################

    def detect(self, image):
        tf.keras.backend.set_learning_phase(0)

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)

        result = self.generator(image, training=False)

        vertices_pred, kp2d_pred, kp3d_pred, pose_pred, shape_pred, cam_pred = result[-1]
        result_dict = {
            "vertices": tf.squeeze(vertices_pred),
            "kp2d": tf.squeeze(kp2d_pred),
            "kp3d": tf.squeeze(kp3d_pred),
            "pose": tf.squeeze(pose_pred),
            "shape": tf.squeeze(shape_pred),
            "cam": tf.squeeze(cam_pred)
        }
        return result_dict


if __name__ == '__main__':
    model = Model()
    result=model.test()
    print(result["psnr"])
    # print(result["kp3d_mpjpe"])
    # print(result["kp3d_mpjpe_aligned"])
    # print(result["bits"])
    # model.train()
